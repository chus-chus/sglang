from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import concurrent.futures
import json
import logging
import math
import time
import os
import threading
from collections import defaultdict
from queue import Empty, Full, PriorityQueue, Queue
from typing import Dict, List, Optional
import datetime

import torch

from sglang.srt.mem_cache.memory_pool import HostKVCache, TokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class LayerDoneCounter:
    def __init__(self, num_layers):
        self.counter = num_layers
        self.condition = threading.Condition()

    def increment(self):
        with self.condition:
            self.counter += 1
            self.condition.notify_all()

    def wait_until(self, threshold):
        with self.condition:
            while self.counter <= threshold:
                self.condition.wait()

    def reset(self):
        with self.condition:
            self.counter = 0


class CacheOperation:

    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    def merge(self, other: "CacheOperation") -> None:
        # multiple operations can be merged into a single operation for batch processing
        self.host_indices = torch.cat([self.host_indices, other.host_indices])
        self.device_indices = torch.cat([self.device_indices, other.device_indices])
        self.priority = min(self.priority, other.priority)
        self.node_ids.extend(other.node_ids)

    def split(self, factor) -> List["CacheOperation"]:
        # split an operation into smaller operations to reduce the size of intermediate buffers
        if factor <= 1:
            return [self]

        chunk_size = math.ceil(len(self.host_indices) / factor)
        split_ops = []
        for i in range(0, len(self.host_indices), chunk_size):
            split_ops.append(
                CacheOperation(
                    host_indices=self.host_indices[i : i + chunk_size],
                    device_indices=self.device_indices[i : i + chunk_size],
                    node_id=0,
                )
            )
        # Inherit the node_ids on the final chunk
        if split_ops:
            split_ops[-1].node_ids = self.node_ids

        return split_ops

    def __lt__(self, other: "CacheOperation"):
        return self.priority < other.priority


class TransferBuffer:
    """
    Overlapping buffer preparation and transfer operations to improve throughput.
    """

    def __init__(
        self, stop_event, buffer_count: int = 3, max_buffer_size: int = 1000
    ) -> None:
        self.stop_event = stop_event
        self.buffers = Queue(maxsize=buffer_count)
        # todo: adjust the buffer size based on throughput profile of the system
        self.max_buffer_size = max_buffer_size

    def full(self) -> bool:
        return self.buffers.full()

    def empty(self) -> bool:
        return self.buffers.empty()

    def put(self, item, block=True, timeout=1) -> None:
        while not self.stop_event.is_set():
            try:
                self.buffers.put(item, block=block, timeout=timeout)
                break
            except Full:
                if not block:
                    break
                continue
            except Exception as e:
                logger.error(e)

    def get(self, block=True, timeout=1) -> Optional[CacheOperation]:
        try:
            return self.buffers.get(block=block, timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(e)

    def clear(self):
        self.buffers.queue.clear()


class CacheTelemetry:
    """
    Track cache hit/miss statistics at both request and block levels.
    """

    _file_lock = threading.Lock()
    _instance_count = 0  # Track number of instances created

    def __init__(self, page_size: int, cache_type: str, output_dir: str, reset_cache_telemetry_on_new_file: bool = False):
        logger.info("[DEBUG] CacheTelemetry: Initializing telemetry tracking")
        CacheTelemetry._instance_count += 1  # Increment the instance counter
        logger.info(f"[DEBUG] CacheTelemetry: Total instances created: {CacheTelemetry._instance_count}")
        self.page_size = page_size
        self.cache_type = cache_type
        self.output_dir = output_dir
        self.reset_cache_telemetry_on_new_file = reset_cache_telemetry_on_new_file # this is a hacky way to reset the counters when the output file is rotated
    
        self.init_fields()
        self.init_time = time.time()
        
    def init_fields(self):
        # block
        self.total_blocks = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0
        self.host_hits = 0
        self.written_blocks_host = 0

        # block (time series)
        self.total_blocks_ts = [] # (timestamp, num_blocks)
        self.total_hits_ts = [] # (timestamp, num_hits)
        self.total_misses_ts = [] # (timestamp, num_misses)
        self.total_evictions_ts = [] # (timestamp, num_evictions)
        self.host_hits_ts = [] # (timestamp, num_hits)
        self.written_blocks_host_ts = [] # (timestamp, num_blocks)
        
        # requests
        self.unique_requests = 0
        self.requests_with_hits = set()
        self.requests_with_misses = set()
        self.requests_with_evictions = set()
        self.tracked_requests = set() # keep track of unique request IDs

        # requests (time series)
        self.unique_requests_ts = [] # (timestamp, num_requests)
        self.requests_with_hits_ts = [] # (timestamp, num_hits)
        self.requests_with_misses_ts = [] # (timestamp, num_misses)
        self.requests_with_evictions_ts = [] # (timestamp, num_evictions)

        self.time_in_write_back = 0

    def reset(self):
        print("[DEBUG] CacheTelemetry: Resetting all telemetry counters")

        # before reset, dump all data, with timestamps included
        if self.total_blocks > 0:
            self.record_stats_ts()
        self.init_fields()
        self.init_time = time.time()

    def record_hit(self, num_blocks: int = 0, request_id=None):
        # print who called this function
        # logger.info(f"[DEBUG] CacheTelemetry hit: req {request_id}, new blocks {num_blocks}, caller {caller}")
        if request_id is not None:
            if request_id not in self.tracked_requests:
                # print(f"[DEBUG] CacheTelemetry: Tracking new request ID: {request_id}")
                self.unique_requests += 1
                self.tracked_requests.add(request_id)
                self.unique_requests_ts.append((time.time() - self.init_time, 1))
                self.requests_with_hits.add(request_id)
                self.requests_with_hits_ts.append((time.time() - self.init_time, 1))
                num_blocks -= 1

                logger.info(f"[DEBUG] CacheTelemetry: New request {request_id} logged new blocks (+{num_blocks})")

        if num_blocks > 0:
            # logger.info(f"[DEBUG] CacheTelemetry: req {request_id} logged new blocks (+{num_blocks})")
            self.total_blocks += num_blocks
            self.total_hits += num_blocks
            # record time series
            timestamp = time.time() - self.init_time
            self.total_blocks_ts.append((timestamp, num_blocks))
            self.total_hits_ts.append((timestamp, num_blocks))

    def record_host_hit(self, num_blocks: int = 0):        
        if num_blocks > 0:
            self.host_hits += num_blocks
            # record time series
            timestamp = time.time() - self.init_time
            self.host_hits_ts.append((timestamp, num_blocks))

    def record_miss(self, num_blocks: int, request_id=None):
        if request_id is not None:
            if request_id not in self.tracked_requests:
                self.unique_requests += 1
                self.tracked_requests.add(request_id)
                self.unique_requests_ts.append((time.time() - self.init_time, 1))
                self.requests_with_misses.add(request_id)
                self.requests_with_misses_ts.append((time.time() - self.init_time, 1))
        
        if num_blocks > 0:
            self.total_blocks += num_blocks
            self.total_misses += num_blocks

            # record time series
            timestamp = time.time() - self.init_time
            self.total_blocks_ts.append((timestamp, num_blocks))
            self.total_misses_ts.append((timestamp, num_blocks))

    def record_eviction(self, num_blocks: int, request_id=None):
        self.total_evictions += num_blocks

        # record time series
        timestamp = time.time() - self.init_time
        self.total_evictions_ts.append((timestamp, num_blocks))

        if request_id is not None:
            if request_id not in self.tracked_requests:
                self.unique_requests += 1
                self.tracked_requests.add(request_id)
                self.unique_requests_ts.append((timestamp, 1))
                self.requests_with_evictions.add(request_id)
                self.requests_with_evictions_ts.append((timestamp, 1))

            # record time series
            self.requests_with_evictions_ts.append((timestamp, num_blocks))

    def record_host_write(self, num_blocks: int):
        self.written_blocks_host += num_blocks
        
        # record time series
        timestamp = time.time() - self.init_time
        self.written_blocks_host_ts.append((timestamp, num_blocks))

    def increment_write_back_time(self, time):
        self.time_in_write_back += time

    def get_all_stats(self) -> Dict:

        return {
            "block_level": {
                "total_blocks": self.total_blocks if self.total_blocks > 0 else 0,
                "hits": self.total_hits if self.total_hits > 0 else 0,
                "misses": self.total_misses,
                "evictions": self.total_evictions,
                "host_hits": self.host_hits,
                "device_hits": self.total_hits - self.host_hits,
                "hit_rate": self.total_hits / self.total_blocks if self.total_blocks > 0 else 0.,
                "miss_rate": self.total_misses / self.total_blocks if self.total_blocks > 0 else 0.,
                "host_hit_rate": self.host_hits / self.total_hits if self.total_hits > 0 else 0.,
                "device_hit_rate": (self.total_hits - self.host_hits) / self.total_hits if self.total_hits > 0 else 0.,
            },
            "request_level": {
                "unique_requests": self.unique_requests,
                "hits": len(self.requests_with_hits),
                "misses": len(self.requests_with_misses),
                "evictions": len(self.requests_with_evictions),
                "hit_rate": len(self.requests_with_hits) / self.unique_requests if self.unique_requests > 0 else 0.,
                "miss_rate": len(self.requests_with_misses) / self.unique_requests if self.unique_requests > 0 else 0.,
            },
            "write_back_time": self.time_in_write_back,
            "cache_type": self.cache_type,
            "page_size": self.page_size,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_all_stats_ts(self) -> Dict:

        return {
            "block_level": {
                "total_blocks": self.total_blocks if self.total_blocks > 0 else 0,
                "hits": self.total_hits if self.total_hits > 0 else 0,
                "misses": self.total_misses,
                "evictions": self.total_evictions,
                "host_hits": self.host_hits,
                "device_hits": self.total_hits - self.host_hits,
                "hit_rate": self.total_hits / self.total_blocks if self.total_blocks > 0 else 0.,
                "miss_rate": self.total_misses / self.total_blocks if self.total_blocks > 0 else 0.,
                "host_hit_rate": self.host_hits / self.total_hits if self.total_hits > 0 else 0.,
                "device_hit_rate": (self.total_hits - self.host_hits) / self.total_hits if self.total_hits > 0 else 0.,
            },
            "block_level_ts": {
                "total_blocks": self.total_blocks_ts,
                "hits": self.total_hits_ts,
                "misses": self.total_misses_ts,
                "evictions": self.total_evictions_ts,
                "host_hits": self.host_hits_ts,
            },
            "request_level_ts": {
                "unique_requests": self.unique_requests_ts,
                "hits": self.requests_with_hits_ts,
                "misses": self.requests_with_misses_ts,
                "evictions": self.requests_with_evictions_ts,
            },
            "request_level": {
                "unique_requests": self.unique_requests,
                "hits": len(self.requests_with_hits),
                "misses": len(self.requests_with_misses),
                "evictions": len(self.requests_with_evictions),
                "hit_rate": len(self.requests_with_hits) / self.unique_requests if self.unique_requests > 0 else 0.,
                "miss_rate": len(self.requests_with_misses) / self.unique_requests if self.unique_requests > 0 else 0.,
            },
            "write_back_time": self.time_in_write_back,
            "cache_type": self.cache_type,
            "page_size": self.page_size,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def record_stats(self):
        # write to disk with safety measures
        stats = self.get_all_stats()
        
        with CacheTelemetry._file_lock:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                
                filepath = os.path.join(self.output_dir, "cache_telemetry.json")

                if not os.path.exists(filepath) and self.reset_cache_telemetry_on_new_file:
                    self.reset()
                    
                with open(filepath, "w") as f:
                    json.dump(stats, f, indent=4)
                    
            except (IOError, OSError) as e:
                logging.warning(f"Failed to write cache telemetry stats: {e}")

    def record_stats_ts(self):
        stats = self.get_all_stats_ts()
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, "cache_telemetry_ts.json")
            with open(filepath, "w", buffering=1024*1024) as f:
                encoder = json.JSONEncoder(indent=4)
                # file can be large, so write in chunks
                for chunk in encoder.iterencode(stats):
                    f.write(chunk)
                    f.flush()         
        except (IOError, OSError) as e:
            logging.warning(f"[CacheTelemetry] Failed to write cache telemetry stats: {e}")
                

class HiCacheController:

    def __init__(
        self,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        load_cache_event: threading.Event = None,
        write_policy: str = "write_through_selective",
    ):
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        self.mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        self.mem_pool_host = mem_pool_host
        self.write_policy = write_policy
        self.page_size = page_size

        self.load_cache_event = load_cache_event
        self.layer_done_counter = LayerDoneCounter(self.mem_pool_device.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        self.write_queue = PriorityQueue()
        self.load_queue = PriorityQueue()

        self.ack_write_queue = Queue()
        self.ack_load_queue = Queue()

        self.stop_event = threading.Event()
        self.write_buffer = TransferBuffer(self.stop_event)
        self.load_buffer = TransferBuffer(
            self.stop_event, buffer_count=10, max_buffer_size=100
        )

        self.write_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

        self.write_thread = threading.Thread(
            target=(
                self.write_thread_func_buffer
                if self.page_size == 1
                else self.write_thread_func_direct
            ),
            daemon=True,
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.write_thread.start()
        self.load_thread.start()

    def reset(self):
        self.stop_event.set()
        self.write_thread.join()
        self.load_thread.join()

        self.write_queue.queue.clear()
        self.load_queue.queue.clear()
        self.write_buffer.clear()
        self.load_buffer.clear()
        self.ack_write_queue.queue.clear()
        self.ack_load_queue.queue.clear()

        self.write_thread = threading.Thread(
            target=(
                self.write_thread_func_buffer
                if self.page_size == 1
                else self.write_thread_func_direct
            ),
            daemon=True,
        )
        self.load_thread = threading.Thread(
            target=self.load_thread_func_layer_by_layer, daemon=True
        )
        self.stop_event.clear()
        self.write_thread.start()
        self.load_thread.start()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.mem_pool_host.protect_write(host_indices)
        self.write_queue.put(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return host_indices

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.mem_pool_host.protect_load(host_indices)
        # to ensure the device indices are ready before accessed by another CUDA stream
        torch.cuda.current_stream().synchronize()
        self.load_queue.put(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def write_thread_func_direct(self):
        """
        Directly write through KV caches to host memory without buffering.
        """
        with torch.cuda.stream(self.write_stream):
            while not self.stop_event.is_set():
                try:
                    operation = self.write_queue.get(block=True, timeout=1)
                    self.mem_pool_host.write_page_all_layers(
                        operation.host_indices,
                        operation.device_indices,
                        self.mem_pool_device,
                    )
                    self.write_stream.synchronize()
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        if node_id != 0:
                            self.ack_write_queue.put(node_id)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(e)

    def load_thread_func_direct(self):
        """
        Directly load KV caches from host memory to device memory without buffering.
        """
        with torch.cuda.stream(self.load_stream):
            while not self.stop_event.is_set():
                try:
                    operation = self.load_queue.get(block=True, timeout=1)
                    # time.sleep(18e-6 * len(operation.host_indices))
                    operation.data = self.mem_pool_host.get_flat_data(
                        operation.host_indices
                    )
                    self.mem_pool_device.transfer(
                        operation.device_indices, operation.data
                    )
                    self.mem_pool_host.complete_io(operation.host_indices)
                    for node_id in operation.node_ids:
                        if node_id != 0:
                            self.ack_load_queue.put(node_id)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(e)

    def load_thread_func_layer_by_layer(self):
        """
        Load KV caches from host memory to device memory layer by layer.
        """
        with torch.cuda.stream(self.load_stream):
            while not self.stop_event.is_set():
                self.load_cache_event.wait(timeout=1)
                if not self.load_cache_event.is_set():
                    continue
                self.load_cache_event.clear()

                batch_operation = None
                while self.load_queue.qsize() > 0:
                    op = self.load_queue.get(block=True)
                    if batch_operation is None:
                        batch_operation = op
                    else:
                        batch_operation.merge(op)
                if batch_operation is None:
                    continue

                self.layer_done_counter.reset()
                for i in range(self.mem_pool_host.layer_num):
                    if self.page_size == 1:
                        flat_data = self.mem_pool_host.get_flat_data_by_layer(
                            batch_operation.host_indices, i
                        )
                        self.mem_pool_device.transfer_per_layer(
                            batch_operation.device_indices, flat_data, i
                        )
                    else:
                        self.mem_pool_host.load_page_per_layer(
                            batch_operation.host_indices,
                            batch_operation.device_indices,
                            self.mem_pool_device,
                            i,
                        )
                        self.load_stream.synchronize()
                    self.layer_done_counter.increment()

                self.mem_pool_host.complete_io(batch_operation.host_indices)
                for node_id in batch_operation.node_ids:
                    if node_id != 0:
                        self.ack_load_queue.put(node_id)

    def write_aux_func(self, no_wait=False):
        """
        Auxiliary function to prepare the buffer for write operations.
        """

        def _to_op(op_):
            assert op_.device_indices.is_cuda, "Device indices should be on GPU"
            op_.data = self.mem_pool_device.get_flat_data(op_.device_indices).to(
                self.mem_pool_host.device
            )
            self.write_buffer.put(op_)
            return op_

        buffer = None
        with torch.cuda.stream(self.write_stream):
            while not self.stop_event.is_set():
                try:
                    operation = self.write_queue.get(block=True, timeout=1)
                    factor = (
                        len(operation.device_indices)
                        // self.write_buffer.max_buffer_size
                    )

                    if factor >= 1:
                        if buffer is not None:
                            _to_op(buffer)
                            buffer = None

                        if factor < 2:
                            _to_op(operation)
                        else:
                            split_ops = operation.split(factor)
                            for op_ in split_ops:
                                _to_op(op_)
                        continue

                    if buffer is None:
                        buffer = operation
                    else:
                        buffer.merge(operation)
                    if (
                        no_wait
                        or len(buffer.host_indices) >= self.write_buffer.max_buffer_size
                        or self.write_queue.empty()
                        or self.write_buffer.empty()
                    ):
                        _to_op(buffer)
                        buffer = None
                except Empty:
                    continue
                except Exception as e:
                    logger.error(e)

    def load_aux_func(self):
        """
        Auxiliary function to prepare the buffer for load operations.
        """

        def _pin_op(op_, put=True):
            op_.data = (
                self.mem_pool_host.get_flat_data(op_.host_indices)
                .contiguous()
                .pin_memory()
            )
            if put:
                self.load_buffer.put(op_)
            return op_

        buffer = None
        while not self.stop_event.is_set():
            try:
                operation = self.load_queue.get(block=True, timeout=1)
                factor = len(operation.host_indices) // self.load_buffer.max_buffer_size

                if factor >= 1:
                    if buffer is not None:
                        _pin_op(buffer)
                        buffer = None

                    if factor < 2:
                        _pin_op(operation)
                    else:
                        split_ops = operation.split(factor)
                        split_args = [(op_, True) for op_ in split_ops[:-1]]
                        split_args.append((split_ops[-1], False))
                        # Spawn threads to pin each op concurrently
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            pinned_ops = list(
                                executor.map(
                                    lambda x: _pin_op(x[0], put=x[1]), split_args
                                )
                            )
                        # preserve the order of last op to ensure correct ack
                        self.load_buffer.put(pinned_ops[-1])
                    continue

                if buffer is None:
                    buffer = operation
                else:
                    buffer.merge(operation)
                if (
                    len(buffer.host_indices) >= self.load_buffer.max_buffer_size
                    or self.load_queue.empty()
                    or self.load_buffer.empty()
                ):
                    _pin_op(buffer)
                    buffer = None
            except Empty:
                continue
            except Exception as e:
                logger.error(e)

    def write_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.write_aux_func, daemon=True)
        aux_thread.start()

        while not self.stop_event.is_set():
            operation = self.write_buffer.get()
            if operation is None:
                continue
            self.mem_pool_host.assign_flat_data(operation.host_indices, operation.data)
            self.mem_pool_host.complete_io(operation.host_indices)
            for node_id in operation.node_ids:
                if node_id != 0:
                    self.ack_write_queue.put(node_id)
        aux_thread.join()

    def load_thread_func_buffer(self):
        aux_thread = threading.Thread(target=self.load_aux_func, daemon=True)
        aux_thread.start()

        with torch.cuda.stream(self.load_stream):
            while not self.stop_event.is_set():
                operation = self.load_buffer.get()
                if operation is None:
                    continue
                self.mem_pool_device.transfer(operation.device_indices, operation.data)
                self.mem_pool_host.complete_io(operation.host_indices)
                for node_id in operation.node_ids:
                    if node_id != 0:
                        self.ack_load_queue.put(node_id)
        aux_thread.join()

    def evict_device(
        self, device_indices: torch.Tensor, host_indices: torch.Tensor
    ) -> int:
        if self.mem_pool_host.is_synced(host_indices):
            self.mem_pool_device_allocator.free(device_indices)
            self.mem_pool_host.update_backup(host_indices)
            return len(device_indices)
        else:
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        if self.mem_pool_host.is_backup(host_indices):
            self.mem_pool_host.free(host_indices)
            return len(host_indices)
        else:
            raise ValueError(
                f"Inconsistent states: {self.mem_pool_host.get_state(host_indices)}"
            )
