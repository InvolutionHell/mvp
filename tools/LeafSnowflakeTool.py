import time

from sentry_sdk.integrations import threading


class LeafSnowflakeBuilder:
    def __init__(self, datacenter_id: int = 1, worker_id: int = 1):
        # 参数范围检查
        if datacenter_id > 31 or datacenter_id < 0:
            raise ValueError("datacenter_id must be between 0 and 31")
        if worker_id > 31 or worker_id < 0:
            raise ValueError("worker_id must be between 0 and 31")

        # 开始时间戳 (2016-11-01)
        self.twepoch = 1477958400000

        # 各部分 bit 位
        self.worker_id_bits = 5
        self.datacenter_id_bits = 5
        self.sequence_bits = 12

        # 最大值
        self.max_worker_id = -1 ^ (-1 << self.worker_id_bits)
        self.max_datacenter_id = -1 ^ (-1 << self.datacenter_id_bits)

        # 位移
        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.timestamp_left_shift = (self.sequence_bits +
                                     self.worker_id_bits +
                                     self.datacenter_id_bits)

        # 序列掩码
        self.sequence_mask = -1 ^ (-1 << self.sequence_bits)

        # 初始化
        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = 0
        self.last_timestamp = -1

        # 锁保证线程安全
        self.lock = threading.Lock()

    def _time_gen(self):
        return int(time.time() * 1000)

    def _til_next_millis(self, last_timestamp):
        timestamp = self._time_gen()
        while timestamp <= last_timestamp:
            timestamp = self._time_gen()
        return timestamp

    def get_id(self):
        with self.lock:
            timestamp = self._time_gen()

            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards. Refusing to generate id")

            if self.last_timestamp == timestamp:
                self.sequence = (self.sequence + 1) & self.sequence_mask
                if self.sequence == 0:
                    timestamp = self._til_next_millis(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            return ((timestamp - self.twepoch) << self.timestamp_left_shift) | \
                   (self.datacenter_id << self.datacenter_id_shift) | \
                   (self.worker_id << self.worker_id_shift) | \
                   self.sequence