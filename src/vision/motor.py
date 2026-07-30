"""
ZDT_X42S/Emm42_5.0 闭环步进电机控制类
兼容 Rock 4D (RK3399) 开发板 / Raspberry Pi / 任何 Linux 系统
基于用户手册 V1.0.5 串口通信协议实现

作者: AI Assistant
版本: 1.0
"""

import serial
import time
import struct
import threading
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import IntEnum, Enum
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmmMotor")


# ============================================================================
# 枚举定义 - 提高代码可读性
# ============================================================================

class MotorState(IntEnum):
    """电机状态标志位 (读取 0x3A 返回)"""
    ENS_TF = 0x01      # 使能状态: 1=已使能
    PRF_TF = 0x02      # 位置到达标志: 1=已到达
    CGI_TF = 0x04      # 堵转标志: 1=检测到堵转
    CGP_TF = 0x08      # 堵转保护标志: 1=已触发保护
    ESI_LF = 0x10      # 左限位开关: 1=高电平
    ESI_RF = 0x20      # 右限位开关: 1=高电平
    OAC_TF = 0x80      # 掉电标志: 1=已掉电


class HomeState(IntEnum):
    """回零状态标志位 (读取 0x3B 返回)"""
    ENC_RDY = 0x01     # 编码器就绪
    CAL_RDY = 0x02     # 校准表就绪
    ORG_SF = 0x04      # 正在回零
    ORG_CF = 0x08      # 回零失败
    OTP_TF = 0x10      # 过热保护
    OCP_TF = 0x20      # 过流保护


class HomeMode(IntEnum):
    """回零模式 (0x9A / 0x4C)"""
    NEAREST = 0x00     # 单圈就近回零
    DIR = 0x01         # 单圈方向回零
    SENLESS = 0x02     # 无限位碰撞回零
    ENDSTOP = 0x03     # 限位开关回零
    ABS_ZERO = 0x04    # 回到坐标原点
    LAST_POS = 0x05    # 回到上次掉电位置


class Direction(IntEnum):
    """电机旋转方向"""
    CW = 0x00          # 顺时针 (正转)
    CCW = 0x01         # 逆时针 (反转)


class MotionMode(IntEnum):
    """位置模式运动模式"""
    REL_PREV = 0x00    # 相对上一输入目标位置
    ABS = 0x01         # 绝对位置 (相对零点)
    REL_CUR = 0x02     # 相对当前位置


class CtrlMode(IntEnum):
    """控制模式"""
    OPEN = 0x00        # 开环模式
    CLOSE = 0x01       # FOC闭环模式


class EnLevel(IntEnum):
    """EN引脚有效电平"""
    LOW = 0x00         # 低电平有效
    HIGH = 0x01        # 高电平有效
    HOLD = 0x02        # 一直使能


class PulPortMode(IntEnum):
    """脉冲端口复用模式"""
    OFF = 0x00         # 关闭
    ENA = 0x01         # 脉冲控制使能
    ESI_RCO = 0x02     # En=限位输入, Dir=到位输出
    LR_ESI = 0x03      # En/Stp=左右限位, Dir=到位输出


class SerialPortMode(IntEnum):
    """通讯端口复用模式"""
    OFF = 0x00         # 关闭
    ESI_ALO = 0x01     # R/A/H=限位输入, T/B/L=报警输出
    UART = 0x02        # 串口通讯 (默认)
    CAN = 0x03         # CAN通讯
    ULR_ESI = 0x04     # 左右限位


class ResponseMode(IntEnum):
    """控制命令应答方式"""
    NONE = 0x00        # 不返回任何命令
    RECEIVE = 0x01     # 只返回确认收到 (默认)
    REACHED = 0x02     # 只返回到位/回零完成
    BOTH = 0x03        # 两者都返回
    OTHER = 0x04       # 位置模式返回完成，其他返回确认


class CheckSumMode(IntEnum):
    """校验方式"""
    FIXED_6B = 0x00    # 固定 0x6B (默认)
    XOR = 0x01         # XOR校验
    CRC8 = 0x02        # CRC8校验
    MODBUS = 0x03      # Modbus-RTU
    DMX512 = 0x04      # DMX512协议


class SysParam(Enum):
    """系统参数读取命令映射"""
    VER = 0x1F          # 固件/硬件版本
    RL = 0x20           # 相电阻/相电感
    PID = 0x21          # PID参数
    VBUS = 0x24         # 总线电压
    CBUS = 0x26         # 总线电流
    CPHA = 0x27         # 相电流
    ENCL = 0x31         # 线性化编码器值
    TPUL = 0x32         # 输入脉冲数
    TPOS = 0x33         # 电机目标位置
    TSET = 0x34         # 实时设定目标位置
    VEL = 0x35          # 实时转速
    CPOS = 0x36         # 实时位置
    PERR = 0x37         # 位置误差
    BAT = 0x38          # 电池电压 (Y42)
    TEMP = 0x39         # 驱动温度
    FLAG = 0x3A         # 电机状态标志
    ORG = 0x3B          # 回零状态标志
    BOTH = 0x3C         # 回零+电机状态 (X42S/Y42)
    PIN10 = 0x3D        # 引脚10电平状态 (X42S/Y42)
    CONF = 0x42         # 驱动配置参数 (需辅助码0x6C)
    STATE = 0x43        # 系统状态参数 (需辅助码0x7A)


# ============================================================================
# CRC8 查表 (用于 CRC8 校验)
# ============================================================================

CRC8_TABLE = [
    0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83,
    0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xF0, 0x1F, 0x41,
    0x9D, 0xC3, 0x21, 0x7F, 0xF0, 0xA2, 0x40, 0x1E,
    0x5F, 0x01, 0xE3, 0xBD, 0x3E, 0x60, 0x82, 0xDC,
    0x23, 0x7D, 0x9F, 0xC1, 0x42, 0x1C, 0xFE, 0xA0,
    0xE1, 0xBF, 0x5D, 0x03, 0x80, 0xDE, 0x3C, 0x62,
    0xBE, 0xE0, 0x02, 0x5C, 0xDF, 0x81, 0x63, 0x3D,
    0x7C, 0x22, 0xC0, 0x9E, 0x1D, 0x43, 0xA1, 0xFF,
    0x46, 0x18, 0xFA, 0xA4, 0x27, 0x79, 0x9B, 0xC5,
    0x84, 0xDA, 0x38, 0x66, 0xE5, 0xBB, 0x59, 0x07,
    0xDB, 0x85, 0x67, 0x39, 0xBA, 0xE4, 0x06, 0x58,
    0x19, 0x47, 0xAE, 0xFB, 0x78, 0x26, 0xC4, 0x9A,
    0x65, 0x3B, 0xD9, 0x87, 0x04, 0x5A, 0xB8, 0xE6,
    0xA7, 0xF9, 0x1B, 0x45, 0xC6, 0x98, 0x7A, 0x24,
    0xF8, 0xA6, 0x44, 0x1A, 0x99, 0xC7, 0x25, 0x7B,
    0x3A, 0x64, 0x86, 0xD8, 0x5B, 0x05, 0xE7, 0xB9,
    0x8C, 0xD2, 0x30, 0x6E, 0xED, 0xB3, 0x51, 0x0F,
    0x4E, 0x10, 0xF2, 0xAC, 0x2F, 0x71, 0x93, 0xCD,
    0x11, 0x4F, 0xAD, 0xF3, 0x70, 0x2E, 0xCC, 0x92,
    0xD3, 0x8D, 0x6F, 0x31, 0xB2, 0xEC, 0x0E, 0x50,
    0xAF, 0xF1, 0x13, 0x4D, 0xCE, 0x90, 0x72, 0x2C,
    0x6D, 0x33, 0xD1, 0x8F, 0x0C, 0x52, 0xB0, 0xEE,
    0x32, 0x6C, 0x8E, 0xD0, 0x53, 0xD0, 0xEF, 0xB1,
    0xF0, 0xAE, 0x4C, 0x12, 0x91, 0xCF, 0x2D, 0x73,
    0xCA, 0x94, 0x76, 0x28, 0xAB, 0xF5, 0x17, 0x49,
    0x08, 0x56, 0xB4, 0xEA, 0x69, 0x37, 0xD5, 0x8B,
    0x57, 0x09, 0xEB, 0xB5, 0x36, 0x68, 0x8A, 0xD4,
    0x95, 0xCB, 0x29, 0x77, 0xF4, 0xAA, 0x48, 0x16,
    0xE9, 0xB7, 0x55, 0x0B, 0x88, 0xD6, 0x34, 0x6A,
    0x2B, 0x75, 0x97, 0xC9, 0x4A, 0x14, 0xF6, 0x8A,
    0x74, 0x2A, 0xC8, 0x96, 0x15, 0x4B, 0xA9, 0xF7,
    0xB6, 0xE8, 0x0A, 0x54, 0xD7, 0x89, 0x6B, 0x35
]


def crc8(data: bytes) -> int:
    """计算 CRC8 校验码"""
    crc = data[0]
    for i in range(1, len(data)):
        crc = CRC8_TABLE[crc ^ data[i]]
    return crc


def xor_checksum(data: bytes) -> int:
    """计算 XOR 校验码"""
    result = 0
    for b in data:
        result ^= b
    return result


# ============================================================================
# 核心电机类
# ============================================================================

class EmmMotor:
    """
    张大头 X42S/Emm42_5.0 闭环步进电机控制类
    
    支持功能:
        - 基础运动: 速度模式、位置模式、力矩模式 (X固件)
        - 回零: 单圈就近、单圈方向、碰撞回零、限位回零
        - 参数读取: 位置、速度、电流、电压、温度、状态标志
        - 参数配置: PID、细分、电流限制、速度限制、地址修改
        - 多机同步: 支持同步标志和广播命令
        - 定时返回: 自动定时返回状态数据
    
    使用示例:
        motor = EmmMotor('/dev/ttyUSB0', motor_id=1)
        motor.enable(True)
        motor.move_to_angle(360, vel=200, acc=100)  # 转一圈
        pos = motor.get_position_angle()
        motor.home(HomeMode.SENLESS)  # 碰撞回零
        motor.disable()
        motor.close()
    """
    
    # 默认参数
    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT = 0.5
    DEFAULT_CHECKSUM = 0x6B
    DEFAULT_PULSE_PER_REV = 3200  # 16细分, 1.8°步进电机
    DEFAULT_ACC = 10  # 默认加速度档位
    
    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        motor_id: int = 1,
        timeout: float = DEFAULT_TIMEOUT,
        pulse_per_rev: int = DEFAULT_PULSE_PER_REV,
        checksum_mode: CheckSumMode = CheckSumMode.FIXED_6B,
        debug: bool = False
    ):
        """
        初始化电机
        
        Args:
            port: 串口设备路径 (如 '/dev/ttyUSB0')
            baudrate: 波特率 (默认 115200)
            motor_id: 电机地址 (1-255, 0为广播地址)
            timeout: 串口超时时间 (秒)
            pulse_per_rev: 每圈脉冲数 (默认 3200, 16细分)
            checksum_mode: 校验方式 (默认 0x6B)
            debug: 是否开启调试日志
        """
        self.port = port
        self.baudrate = baudrate
        self.motor_id = motor_id
        self.timeout = timeout
        self.pulse_per_rev = pulse_per_rev
        self.checksum_mode = checksum_mode
        self.debug = debug
        
        # 串口对象
        self._serial: Optional[serial.Serial] = None
        
        # 定时返回线程
        self._timer_running = False
        self._timer_thread: Optional[threading.Thread] = None
        self._timer_callback: Optional[callable] = None
        self._timer_param = None
        
        # 锁 (用于多线程安全)
        self._lock = threading.Lock()
        
        # 初始化串口
        self._init_serial()
        
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"电机初始化完成: ID={motor_id}, 端口={port}, 每圈脉冲={pulse_per_rev}")
    
    # ========================================================================
    # 串口底层操作
    # ========================================================================
    
    def _init_serial(self):
        """初始化串口"""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            if self._serial.is_open:
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
            logger.debug(f"串口打开成功: {self.port}")
        except Exception as e:
            raise RuntimeError(f"打开串口 {self.port} 失败: {e}")
    
    def _build_command(self, addr: int, func: int, data: bytes = b'') -> bytes:
        """
        构建命令包 (带校验)
        
        Args:
            addr: 电机地址
            func: 功能码
            data: 数据部分
            
        Returns:
            完整的命令字节串
        """
        cmd = bytes([addr, func]) + data
        
        if self.checksum_mode == CheckSumMode.FIXED_6B:
            cmd += b'\x6B'
        elif self.checksum_mode == CheckSumMode.XOR:
            cmd += bytes([xor_checksum(cmd)])
        elif self.checksum_mode == CheckSumMode.CRC8:
            cmd += bytes([crc8(cmd)])
        else:
            cmd += b'\x6B'  # 默认
        
        return cmd
    
    def _send_command(self, cmd: bytes, wait_response: bool = True,
                      response_len: int = 4, timeout: float = None) -> Optional[bytes]:
        """
        发送命令并接收响应

        Args:
            cmd: 命令字节串
            wait_response: 是否等待响应
            response_len: 期望响应长度 (字节)
            timeout: 超时时间 (秒)

        Returns:
            响应数据 或 None
        """
        with self._lock:
            if not self._serial or not self._serial.is_open:
                raise RuntimeError("串口未打开")

            # 清空输入缓冲区
            self._serial.reset_input_buffer()

            # 发送
            if self.debug:
                logger.debug(f"发送: {cmd.hex().upper()}")
            self._serial.write(cmd)
            self._serial.flush()

            if not wait_response:
                return None

            timeout = timeout or self.timeout
            echo_len = len(cmd)

            # ---- 第1步：吞咽回显 (echo) ----
            # Emm42 V5 在回显模式下会把命令原样返回。
            # 回显字节数 = 命令字节数，必须先吞掉再读真实响应。
            echo_buf = bytearray()
            echo_deadline = time.time() + min(timeout, 0.05)
            while len(echo_buf) < echo_len and time.time() < echo_deadline:
                if self._serial.in_waiting:
                    echo_buf.extend(self._serial.read(self._serial.in_waiting))
                else:
                    time.sleep(0.0005)

            if self.debug and echo_buf:
                logger.debug(f"回显: {echo_buf.hex().upper()}")

            # ---- 第2步：读取真实响应 ----
            start = time.time()
            response = bytearray()

            while time.time() - start < timeout:
                if self._serial.in_waiting:
                    data = self._serial.read(self._serial.in_waiting)
                    response.extend(data)

                    # 检查是否收到足够数据
                    if len(response) >= response_len:
                        break
                else:
                    time.sleep(0.0005)

            if self.debug and len(response) > 0:
                logger.debug(f"接收: {response.hex().upper()}")

            return bytes(response) if response else None
    
    def _check_response(self, response: bytes, addr: int, func: int) -> Tuple[bool, int]:
        """
        检查响应是否正确
        
        Returns:
            (是否成功, 错误码)
            错误码: 0=成功, 0xE2=参数错误, 0xEE=格式错误, 0x12/0x22=回零特殊状态
        """
        if not response or len(response) < 3:
            return False, -1
        
        if response[0] != addr:
            return False, -1
        
        status = response[2] if len(response) > 2 else 0xFF
        
        if status == 0x02:
            return True, 0
        elif status == 0xE2:
            return False, 0xE2
        elif status == 0xEE:
            return False, 0xEE
        elif status in (0x12, 0x22):
            # 回零特殊状态 (已在零点/限位触发)
            return True, status
        
        return False, status
    
    # ========================================================================
    # 触发动作命令 (5.2)
    # ========================================================================
    
    def calibrate_encoder(self, addr: int = None) -> bool:
        """
        触发编码器校准 (5.2.1)
        电机将缓慢正转一圈然后反转一圈进行线性化校准
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x06, b'\x45')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x06)
        return success
    
    def restart_motor(self, addr: int = None) -> bool:
        """
        重启电机 (5.2.2)
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x08, b'\x97')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x08)
        return success
    
    def reset_position_to_zero(self, addr: int = None) -> bool:
        """
        将当前位置角度清零 (5.2.3)
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x0A, b'\x6D')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x0A)
        return success
    
    def clear_protection(self, addr: int = None) -> bool:
        """
        解除堵转/过热/过流保护 (5.2.4)
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x0E, b'\x52')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x0E)
        return success
    
    def factory_reset(self, addr: int = None) -> bool:
        """
        恢复出厂设置 (5.2.5)
        注意: 恢复后需要断电重新上电，并重新校准编码器
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x0F, b'\x5F')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x0F)
        return success
    
    # ========================================================================
    # 运动控制命令 (5.3)
    # ========================================================================
    
    def enable(self, state: bool = True, sync: bool = False, addr: int = None) -> bool:
        """
        电机使能控制 (5.3.2)
        
        Args:
            state: True=使能(锁轴), False=去使能(松轴)
            sync: True=缓存命令等待同步触发, False=立即执行
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        data = bytes([0xAB, 0x01 if state else 0x00, 0x01 if sync else 0x00])
        cmd = self._build_command(addr, 0xF3, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xF3)
        return success
    
    # ---- 速度模式 ----
    
    def set_velocity(self, rpm: float, direction: Direction = Direction.CW, 
                     acc: int = DEFAULT_ACC, sync: bool = False, addr: int = None) -> bool:
        """
        速度模式控制 (Emm固件) (5.3.7)
        
        Args:
            rpm: 目标转速 (0-3000 RPM)
            direction: 旋转方向 (CW/CCW)
            acc: 加速度档位 (0-255)
                 0=直接启动, 值越大加速越快
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        vel = int(max(0, min(3000, rpm)))
        acc = max(0, min(255, acc))
        
        data = bytes([
            direction & 0xFF,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            acc & 0xFF,
            0x01 if sync else 0x00
        ])
        cmd = self._build_command(addr, 0xF6, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xF6)
        return success
    
    # ---- 位置模式 ----
    
    def move_to_pulse(self, pulses: int, rpm: float = 200, acc: int = DEFAULT_ACC,
                      mode: MotionMode = MotionMode.REL_PREV, sync: bool = False,
                      addr: int = None) -> bool:
        """
        位置模式控制 (Emm固件) (5.3.12)
        使用脉冲数作为位置单位
        
        Args:
            pulses: 目标脉冲数 (int32, 可正可负)
                    >0: CW方向, <0: CCW方向
            rpm: 速度 (0-3000 RPM)
            acc: 加速度档位 (0-255)
            mode: 运动模式 (REL_PREV/ABS/REL_CUR)
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        rpm = int(max(0, min(3000, rpm)))
        acc = max(0, min(255, acc))
        
        # 处理方向
        direction = Direction.CW
        pulse_abs = pulses
        if pulses < 0:
            direction = Direction.CCW
            pulse_abs = -pulses
        
        pulse_abs = max(0, min(0xFFFFFFFF, pulse_abs))
        
        data = bytes([
            direction & 0xFF,
            (rpm >> 8) & 0xFF,
            rpm & 0xFF,
            acc & 0xFF,
            (pulse_abs >> 24) & 0xFF,
            (pulse_abs >> 16) & 0xFF,
            (pulse_abs >> 8) & 0xFF,
            pulse_abs & 0xFF,
            mode & 0xFF,
            0x01 if sync else 0x00
        ])
        cmd = self._build_command(addr, 0xFD, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xFD)
        return success
    
    def move_to_angle(self, angle_deg: float, rpm: float = 200, acc: int = DEFAULT_ACC,
                      mode: MotionMode = MotionMode.REL_PREV, sync: bool = False,
                      addr: int = None) -> bool:
        """
        按角度移动 (Emm固件)
        
        Args:
            angle_deg: 目标角度 (度), 可正可负
            rpm: 速度 (0-3000 RPM)
            acc: 加速度档位 (0-255)
            mode: 运动模式
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        # 角度转脉冲数
        pulses = int(angle_deg * self.pulse_per_rev / 360.0)
        return self.move_to_pulse(pulses, rpm, acc, mode, sync, addr)
    
    # ---- 快速位置模式 (Emm V2.0+) ----
    
    def config_fast_position(self, rpm: float = 200, acc: int = DEFAULT_ACC,
                             mode: MotionMode = MotionMode.REL_PREV,
                             sync: bool = False, addr: int = None) -> bool:
        """
        配置快速位置模式参数 (5.3.13)
        设定后，后续只需发送位置值即可运动
        
        Args:
            rpm: 速度 (0-3000 RPM)
            acc: 加速度档位 (0-255)
            mode: 运动模式
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        rpm = int(max(0, min(3000, rpm)))
        acc = max(0, min(255, acc))
        
        data = bytes([
            (rpm >> 8) & 0xFF,
            rpm & 0xFF,
            acc & 0xFF,
            mode & 0xFF,
            0x01 if sync else 0x00
        ])
        cmd = self._build_command(addr, 0xF1, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xF1)
        return success
    
    def fast_move_pulse(self, pulses: int, addr: int = None) -> bool:
        """
        快速位置模式 - 发送脉冲数 (5.3.13)
        需要先调用 config_fast_position 配置参数
        
        Args:
            pulses: 目标脉冲数 (int32, 可正可负)
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        pulse_u32 = pulses & 0xFFFFFFFF
        
        data = bytes([
            (pulse_u32 >> 24) & 0xFF,
            (pulse_u32 >> 16) & 0xFF,
            (pulse_u32 >> 8) & 0xFF,
            pulse_u32 & 0xFF
        ])
        cmd = self._build_command(addr, 0xFC, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xFC)
        return success
    
    def fast_move_angle(self, angle_deg: float, addr: int = None) -> bool:
        """
        快速位置模式 - 按角度移动
        需要先调用 config_fast_position 配置参数
        
        Args:
            angle_deg: 目标角度 (度), 可正可负
            addr: 电机地址
            
        Returns:
            是否成功
        """
        pulses = int(angle_deg * self.pulse_per_rev / 360.0)
        return self.fast_move_pulse(pulses, addr)
    
    # ---- 立即停止 ----
    
    def stop(self, sync: bool = False, addr: int = None) -> bool:
        """
        立即停止电机 (5.3.15)
        
        Args:
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        data = bytes([0x98, 0x01 if sync else 0x00])
        cmd = self._build_command(addr, 0xFE, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xFE)
        return success
    
    # ---- 多机同步 ----
    
    def trigger_sync(self, addr: int = 0) -> bool:
        """
        触发多机同步运动 (5.3.16)
        使用广播地址 (0) 让所有缓存了命令的电机同步执行
        
        Args:
            addr: 通常用广播地址 0
            
        Returns:
            是否成功
        """
        cmd = self._build_command(addr, 0xFF, b'\x66')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xFF)
        return success
    
    # ---- 多电机命令 (X42S/Y42) ----
    
    def send_multi_motor_command(self, commands: List[bytes], addr: int = 0) -> bool:
        """
        发送多电机命令 (5.3.1)
        一次性发送多个电机的不同命令
        
        Args:
            commands: 各电机的命令字节串列表
            addr: 广播地址 (默认 0)
            
        Returns:
            是否成功
        """
        # 计算总字节数
        total_len = 2  # AA + 长度字节
        for cmd in commands:
            total_len += len(cmd)
        
        data = bytearray([0xAA, total_len & 0xFF])
        for cmd in commands:
            data.extend(cmd)
        data.append(0x6B)  # 校验
        
        # 发送 (不等待响应, 避免总线冲突)
        self._send_command(bytes(data), wait_response=False)
        return True
    
    # ========================================================================
    # 回零命令 (5.4)
    # ========================================================================
    
    def set_home_position(self, save: bool = True, addr: int = None) -> bool:
        """
        设置单圈回零的零点位置 (5.4.1)
        将当前位置设为零点
        
        Args:
            save: True=存储到Flash掉电不丢失
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        data = bytes([0x88, 0x01 if save else 0x00])
        cmd = self._build_command(addr, 0x93, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x93)
        return success
    
    def trigger_home(self, mode: HomeMode = HomeMode.NEAREST, 
                     sync: bool = False, addr: int = None) -> bool:
        """
        触发回零 (5.4.2)
        
        Args:
            mode: 回零模式
            sync: True=缓存命令等待同步触发
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        data = bytes([mode & 0xFF, 0x01 if sync else 0x00])
        cmd = self._build_command(addr, 0x9A, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x9A)
        return success
    
    def abort_home(self, addr: int = None) -> bool:
        """
        强制中断并退出回零操作 (5.4.3)
        
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x9C, b'\x48')
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x9C)
        return success
    
    def get_home_status(self, addr: int = None) -> Optional[Dict[str, bool]]:
        """
        读取回零状态标志 (5.4.4)
        
        Returns:
            字典包含: enc_ready, cal_ready, homing, home_failed, 
                      over_temp, over_current
            或 None (读取失败)
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x3B, b'')
        resp = self._send_command(cmd, response_len=4)
        
        if not resp or len(resp) < 3:
            return None
        
        if resp[0] != addr or resp[1] != 0x3B:
            return None
        
        status = resp[2]
        return {
            'enc_ready': bool(status & 0x01),
            'cal_ready': bool(status & 0x02),
            'homing': bool(status & 0x04),
            'home_failed': bool(status & 0x08),
            'over_temp': bool(status & 0x10),
            'over_current': bool(status & 0x20),
        }
    
    def get_home_params(self, addr: int = None) -> Optional[Dict[str, Any]]:
        """
        读取回零参数 (5.4.5)
        
        Returns:
            字典包含: mode, direction, speed, timeout, 
                      collision_rpm, collision_ma, collision_ms, auto_enable
            或 None (读取失败)
        """
        addr = addr or self.motor_id
        cmd = self._build_command(addr, 0x22, b'')
        resp = self._send_command(cmd, response_len=18)
        
        if not resp or len(resp) < 18:
            return None
        
        if resp[0] != addr or resp[1] != 0x22:
            return None
        
        return {
            'mode': resp[2],
            'direction': resp[3],
            'speed': (resp[4] << 8) | resp[5],
            'timeout': (resp[6] << 24) | (resp[7] << 16) | (resp[8] << 8) | resp[9],
            'collision_rpm': (resp[10] << 8) | resp[11],
            'collision_ma': (resp[12] << 8) | resp[13],
            'collision_ms': (resp[14] << 8) | resp[15],
            'auto_enable': resp[16] == 1,
        }
    
    def set_home_params(
        self,
        mode: HomeMode = HomeMode.NEAREST,
        direction: Direction = Direction.CW,
        speed_rpm: int = 30,
        timeout_ms: int = 10000,
        collision_rpm: int = 300,
        collision_ma: int = 800,
        collision_ms: int = 60,
        auto_enable: bool = False,
        save: bool = True,
        addr: int = None
    ) -> bool:
        """
        修改回零参数 (5.4.6)
        
        Args:
            mode: 回零模式
            direction: 回零方向
            speed_rpm: 回零速度 (0-3000 RPM)
            timeout_ms: 超时时间 (毫秒)
            collision_rpm: 碰撞检测转速
            collision_ma: 碰撞检测电流 (mA)
            collision_ms: 碰撞检测时间 (毫秒)
            auto_enable: 是否上电自动触发回零
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        speed = max(0, min(0x0BBB, speed_rpm))
        timeout = max(0, min(0xFFFFFFFF, timeout_ms))
        col_rpm = max(0, min(0x0BBB, collision_rpm))
        col_ma = max(0, min(0x1388, collision_ma))
        col_ms = max(0, min(0xFFFF, collision_ms))
        
        data = bytes([
            0xAE,
            0x01 if save else 0x00,
            mode & 0xFF,
            direction & 0xFF,
            (speed >> 8) & 0xFF,
            speed & 0xFF,
            (timeout >> 24) & 0xFF,
            (timeout >> 16) & 0xFF,
            (timeout >> 8) & 0xFF,
            timeout & 0xFF,
            (col_rpm >> 8) & 0xFF,
            col_rpm & 0xFF,
            (col_ma >> 8) & 0xFF,
            col_ma & 0xFF,
            (col_ms >> 8) & 0xFF,
            col_ms & 0xFF,
            0x01 if auto_enable else 0x00
        ])
        cmd = self._build_command(addr, 0x4C, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x4C)
        return success
    
    # ========================================================================
    # 读取系统参数 (5.5)
    # ========================================================================
    
    def _read_sys_param(self, param: SysParam, addr: int = None,
                        extra: bytes = b'') -> Optional[bytes]:
        """读取系统参数 (底层)"""
        addr = addr or self.motor_id
        cmd = self._build_command(addr, param.value, extra)

        # 不同参数响应长度不同，给一个合理的预期值
        _RESP_LEN = {
            SysParam.FLAG: 4, SysParam.ORG: 4,
            SysParam.VBUS: 5, SysParam.CBUS: 5, SysParam.CPHA: 5,
            SysParam.ENCL: 5, SysParam.VEL: 5, SysParam.TEMP: 5, SysParam.BAT: 5,
            SysParam.VER: 7, SysParam.RL: 7, SysParam.TPUL: 7,
            SysParam.TPOS: 7, SysParam.TSET: 7,
            SysParam.CPOS: 7, SysParam.PERR: 7,
            SysParam.PID: 15,
        }
        resp_len = _RESP_LEN.get(param, 8)
        # 响应读取超时 30ms（电机回显后很快回复）
        resp = self._send_command(cmd, response_len=resp_len, timeout=0.03)

        if not resp or len(resp) < 3:
            return None
        if resp[0] != addr or resp[1] != param.value:
            return None

        return resp
    
    def get_version(self, addr: int = None) -> Optional[Dict[str, Any]]:
        """
        读取固件版本和硬件版本 (5.5.2)
        
        Returns:
            字典: firmware_ver, hw_series, hw_type, hw_ver
        """
        resp = self._read_sys_param(SysParam.VER, addr)
        if not resp or len(resp) < 7:
            return None
        
        fw = (resp[2] << 8) | resp[3]
        hw = (resp[4] << 8) | resp[5]
        
        series_map = {0: 'X', 1: 'Y'}
        type_map = {0: 20, 1: 28, 2: 35, 3: 42, 4: 57, 5: 86}
        
        return {
            'firmware_ver': fw / 100.0,
            'hw_series': series_map.get((hw >> 12) & 0x0F, 'Unknown'),
            'hw_type': type_map.get((hw >> 8) & 0x0F, 0),
            'hw_ver': (hw & 0xFF) / 10.0,
        }
    
    def get_bus_voltage(self, addr: int = None) -> Optional[float]:
        """
        读取总线电压 (5.5.4)
        
        Returns:
            电压值 (V), 或 None
        """
        resp = self._read_sys_param(SysParam.VBUS, addr)
        if not resp or len(resp) < 5:
            return None
        return ((resp[2] << 8) | resp[3]) / 1000.0
    
    def get_bus_current(self, addr: int = None) -> Optional[float]:
        """
        读取总线电流 (5.5.5) (X42S/Y42)
        
        Returns:
            电流值 (A), 或 None
        """
        resp = self._read_sys_param(SysParam.CBUS, addr)
        if not resp or len(resp) < 5:
            return None
        return ((resp[2] << 8) | resp[3]) / 1000.0
    
    def get_phase_current(self, addr: int = None) -> Optional[float]:
        """
        读取相电流 (5.5.6)
        
        Returns:
            电流值 (A), 或 None
        """
        resp = self._read_sys_param(SysParam.CPHA, addr)
        if not resp or len(resp) < 5:
            return None
        return ((resp[2] << 8) | resp[3]) / 1000.0
    
    def get_encoder_value(self, addr: int = None) -> Optional[int]:
        """
        读取线性化编码器值 (5.5.7)
        范围 0-65535 对应 0-360度
        
        Returns:
            编码器值 (0-65535), 或 None
        """
        resp = self._read_sys_param(SysParam.ENCL, addr)
        if not resp or len(resp) < 5:
            return None
        return (resp[2] << 8) | resp[3]
    
    def get_input_pulses(self, addr: int = None) -> Optional[int]:
        """
        读取输入脉冲数 (5.5.8)
        
        Returns:
            脉冲数 (可正可负), 或 None
        """
        resp = self._read_sys_param(SysParam.TPUL, addr)
        if not resp or len(resp) < 7:
            return None
        
        pulses = (resp[3] << 24) | (resp[4] << 16) | (resp[5] << 8) | resp[6]
        if resp[2] != 0:
            pulses = -pulses
        return pulses
    
    def get_target_position(self, addr: int = None) -> Optional[float]:
        """
        读取电机目标位置 (5.5.9)
        Emm固件: 0-65535 对应 0-360度
        
        Returns:
            目标位置 (度), 或 None
        """
        resp = self._read_sys_param(SysParam.TPOS, addr)
        if not resp or len(resp) < 7:
            return None
        
        pos = (resp[3] << 24) | (resp[4] << 16) | (resp[5] << 8) | resp[6]
        # Emm固件: 0-65535 对应 0-360度
        angle = pos * 360.0 / 65536.0
        if resp[2] != 0:
            angle = -angle
        return angle
    
    def get_velocity(self, addr: int = None) -> Optional[float]:
        """
        读取实时转速 (5.5.11)
        
        Returns:
            转速 (RPM), 或 None
        """
        resp = self._read_sys_param(SysParam.VEL, addr)
        if not resp or len(resp) < 5:
            return None
        
        vel = (resp[3] << 8) | resp[4]
        if resp[2] != 0:
            vel = -vel
        return vel / 10.0  # X固件需要/10, Emm固件直接返回
    
    def get_position(self, addr: int = None) -> Optional[float]:
        """
        读取实时位置 (5.5.13)
        Emm固件: 0-65535 对应 0-360度
        
        Returns:
            位置 (度), 或 None
        """
        resp = self._read_sys_param(SysParam.CPOS, addr)
        if not resp or len(resp) < 7:
            return None
        
        pos = (resp[3] << 24) | (resp[4] << 16) | (resp[5] << 8) | resp[6]
        # Emm固件: 0-65535 对应 0-360度
        angle = pos * 360.0 / 65536.0
        if resp[2] != 0:
            angle = -angle
        return angle
    
    def get_position_error(self, addr: int = None) -> Optional[float]:
        """
        读取位置误差 (5.5.14)
        Emm固件: 0-65535 对应 0-360度
        
        Returns:
            位置误差 (度), 或 None
        """
        resp = self._read_sys_param(SysParam.PERR, addr)
        if not resp or len(resp) < 7:
            return None
        
        err = (resp[3] << 24) | (resp[4] << 16) | (resp[5] << 8) | resp[6]
        angle = err * 360.0 / 65536.0
        if resp[2] != 0:
            angle = -angle
        return angle
    
    def get_temperature(self, addr: int = None) -> Optional[float]:
        """
        读取驱动温度 (5.5.12) (X42S/Y42)
        
        Returns:
            温度 (℃), 或 None
        """
        resp = self._read_sys_param(SysParam.TEMP, addr)
        if not resp or len(resp) < 5:
            return None
        
        temp = (resp[3] << 8) | resp[4]
        if resp[2] != 0:
            temp = -temp
        return temp
    
    def get_status_flags(self, addr: int = None) -> Optional[Dict[str, bool]]:
        """
        读取电机状态标志 (5.5.15)
        
        Returns:
            字典: enabled, position_reached, stall, stall_protected,
                  left_limit, right_limit, power_lost
            或 None
        """
        resp = self._read_sys_param(SysParam.FLAG, addr)
        if not resp or len(resp) < 4:
            return None
        
        flags = resp[2]
        return {
            'enabled': bool(flags & MotorState.ENS_TF),
            'position_reached': bool(flags & MotorState.PRF_TF),
            'stall': bool(flags & MotorState.CGI_TF),
            'stall_protected': bool(flags & MotorState.CGP_TF),
            'left_limit': bool(flags & MotorState.ESI_LF),
            'right_limit': bool(flags & MotorState.ESI_RF),
            'power_lost': bool(flags & MotorState.OAC_TF),
        }
    
    def get_battery_voltage(self, addr: int = None) -> Optional[float]:
        """
        读取电池电压 (5.5.18) (Y42 掉电记录)
        
        Returns:
            电压值 (V), 或 None
        """
        resp = self._read_sys_param(SysParam.BAT, addr)
        if not resp or len(resp) < 5:
            return None
        return ((resp[2] << 8) | resp[3]) / 1000.0
    
    # ========================================================================
    # 读写驱动参数 (5.6)
    # ========================================================================
    
    def set_motor_id(self, new_id: int, save: bool = True, addr: int = None) -> bool:
        """
        修改电机ID/地址 (5.6.1)
        
        Args:
            new_id: 新地址 (1-255)
            save: 是否保存到Flash
            addr: 当前地址 (如果不知道, 可以用广播0)
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        new_id = max(1, min(255, new_id))
        data = bytes([0x4B, 0x01 if save else 0x00, new_id & 0xFF])
        cmd = self._build_command(addr, 0xAE, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xAE)
        if success:
            self.motor_id = new_id
        return success
    
    def set_microstep(self, step: int, save: bool = True, addr: int = None) -> bool:
        """
        修改细分值 (5.6.2)
        
        Args:
            step: 细分值 (1-256, 256用0表示)
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        step_val = 0 if step == 256 else step
        data = bytes([0x8A, 0x01 if save else 0x00, step_val & 0xFF])
        cmd = self._build_command(addr, 0x84, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x84)
        if success:
            self.pulse_per_rev = 3200 * step / 16  # 更新脉冲数
        return success
    
    def set_open_loop_current(self, current_ma: int, save: bool = True, addr: int = None) -> bool:
        """
        修改开环模式工作电流 (5.6.12)
        
        Args:
            current_ma: 电流 (0-5000 mA)
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        current = max(0, min(5000, current_ma))
        data = bytes([0x33, 0x01 if save else 0x00, (current >> 8) & 0xFF, current & 0xFF])
        cmd = self._build_command(addr, 0x44, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x44)
        return success
    
    def set_closed_loop_current(self, current_ma: int, save: bool = True, addr: int = None) -> bool:
        """
        修改闭环模式最大电流 (5.6.13)
        
        Args:
            current_ma: 电流 (0-5000 mA)
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        current = max(0, min(5000, current_ma))
        data = bytes([0x66, 0x01 if save else 0x00, (current >> 8) & 0xFF, current & 0xFF])
        cmd = self._build_command(addr, 0x45, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x45)
        return success
    
    def set_pid_params(self, kp: int, ki: int, kd: int, save: bool = True, addr: int = None) -> bool:
        """
        修改PID参数 (Emm固件) (5.6.17)
        
        Args:
            kp: 比例系数
            ki: 积分系数
            kd: 微分系数
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        data = bytes([
            0xC3,
            0x01 if save else 0x00,
            (kp >> 24) & 0xFF,
            (kp >> 16) & 0xFF,
            (kp >> 8) & 0xFF,
            kp & 0xFF,
            (ki >> 24) & 0xFF,
            (ki >> 16) & 0xFF,
            (ki >> 8) & 0xFF,
            ki & 0xFF,
            (kd >> 24) & 0xFF,
            (kd >> 16) & 0xFF,
            (kd >> 8) & 0xFF,
            kd & 0xFF,
        ])
        cmd = self._build_command(addr, 0x4A, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x4A)
        return success
    
    def get_pid_params(self, addr: int = None) -> Optional[Dict[str, int]]:
        """
        读取PID参数 (Emm固件) (5.6.16)
        
        Returns:
            字典: kp, ki, kd
        """
        resp = self._read_sys_param(SysParam.PID, addr)
        if not resp or len(resp) < 15:
            return None
        
        return {
            'kp': (resp[2] << 24) | (resp[3] << 16) | (resp[4] << 8) | resp[5],
            'ki': (resp[6] << 24) | (resp[7] << 16) | (resp[8] << 8) | resp[9],
            'kd': (resp[10] << 24) | (resp[11] << 16) | (resp[12] << 8) | resp[13],
        }
    
    def set_heartbeat_timeout(self, timeout_ms: int, save: bool = True, addr: int = None) -> bool:
        """
        修改心跳保护功能时间 (5.6.25)
        在设定时间内没有收到命令，电机将急停
        
        Args:
            timeout_ms: 超时时间 (毫秒), 0=关闭
            save: 是否保存到Flash
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        timeout = max(0, min(0xFFFFFFFF, timeout_ms))
        data = bytes([
            0x38,
            0x01 if save else 0x00,
            (timeout >> 24) & 0xFF,
            (timeout >> 16) & 0xFF,
            (timeout >> 8) & 0xFF,
            timeout & 0xFF,
        ])
        cmd = self._build_command(addr, 0x68, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x68)
        return success
    
    # ========================================================================
    # 上电自动运行 (5.7)
    # ========================================================================
    
    def set_auto_run(self, rpm: float, direction: Direction = Direction.CW,
                     acc: int = DEFAULT_ACC, enable_pin: bool = False,
                     save: bool = True, addr: int = None) -> bool:
        """
        存储上电自动运行参数 (Emm固件) (5.7.2)
        
        Args:
            rpm: 速度 (0-3000 RPM)
            direction: 方向
            acc: 加速度档位 (0-255)
            enable_pin: 是否使能En引脚控制启停
            save: True=存储, False=清除
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        vel = int(max(0, min(3000, rpm)))
        acc = max(0, min(255, acc))
        
        data = bytes([
            0x1C,
            0x01 if save else 0x00,
            direction & 0xFF,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            acc & 0xFF,
            0x01 if enable_pin else 0x00,
        ])
        cmd = self._build_command(addr, 0xF7, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0xF7)
        return success
    
    # ========================================================================
    # 定时返回信息 (5.5.1)
    # ========================================================================
    
    def start_timer_return(self, param: SysParam, interval_ms: int,
                           callback: callable = None, addr: int = None) -> bool:
        """
        启动定时返回信息 (5.5.1)
        
        Args:
            param: 要返回的参数类型 (SysParam)
            interval_ms: 定时时间 (毫秒), 0=停止
            callback: 回调函数 (接收返回数据)
            addr: 电机地址
            
        Returns:
            是否成功
        """
        addr = addr or self.motor_id
        interval = max(0, min(0xFFFF, interval_ms))
        
        data = bytes([
            0x18,
            param.value & 0xFF,
            (interval >> 8) & 0xFF,
            interval & 0xFF,
        ])
        cmd = self._build_command(addr, 0x11, data)
        resp = self._send_command(cmd, response_len=4)
        success, _ = self._check_response(resp, addr, 0x11)
        
        if success and interval > 0:
            self._timer_running = True
            self._timer_callback = callback
            self._timer_param = (addr, param)
        else:
            self._timer_running = False
        
        return success
    
    def stop_timer_return(self, addr: int = None) -> bool:
        """停止定时返回"""
        return self.start_timer_return(SysParam.FLAG, 0, None, addr)
    
    # ========================================================================
    # 便捷工具函数
    # ========================================================================
    
    def wait_for_position(self, target_angle: float, tolerance: float = 0.5,
                          timeout: float = 10.0, addr: int = None) -> bool:
        """
        等待电机到达目标位置
        
        Args:
            target_angle: 目标角度 (度)
            tolerance: 允许误差 (度)
            timeout: 超时时间 (秒)
            addr: 电机地址
            
        Returns:
            是否到达
        """
        addr = addr or self.motor_id
        start = time.time()
        
        while time.time() - start < timeout:
            pos = self.get_position(addr)
            if pos is None:
                time.sleep(0.01)
                continue
            
            error = abs(pos - target_angle)
            if error < tolerance:
                return True
            
            # 检查是否堵转
            flags = self.get_status_flags(addr)
            if flags and flags.get('stall_protected', False):
                return False
            
            time.sleep(0.01)
        
        return False
    
    def wait_for_stop(self, timeout: float = 5.0, addr: int = None) -> bool:
        """
        等待电机停止
        
        Args:
            timeout: 超时时间 (秒)
            addr: 电机地址
            
        Returns:
            是否停止
        """
        addr = addr or self.motor_id
        start = time.time()
        
        while time.time() - start < timeout:
            vel = self.get_velocity(addr)
            if vel is None:
                time.sleep(0.01)
                continue
            
            if abs(vel) < 0.5:
                return True
            
            time.sleep(0.01)
        
        return False
    
    # ========================================================================
    # 资源管理
    # ========================================================================
    
    def close(self):
        """关闭串口"""
        self._timer_running = False
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=1.0)
        
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("串口已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建电机实例
    motor = EmmMotor(
        port='/dev/ttyACM0',  # 请根据实际设备修改
        motor_id=1,
        debug=True
    )
    
    try:
        # 1. 获取版本信息
        ver = motor.get_version()
        print(f"固件版本: {ver}")
        
        # 2. 使能电机
        motor.enable(True)
        
        # 3. 读取状态
        flags = motor.get_status_flags()
        print(f"状态: {flags}")
        
        # 4. 位置清零
        motor.reset_position_to_zero()
        
        # 5. 移动电机: 转一圈 (相对位置)
        motor.move_to_angle(360.0, rpm=200, acc=50)
        
        # 6. 等待到达
        motor.wait_for_position(360.0)
        
        # 7. 获取当前位置
        pos = motor.get_position()
        if pos is not None:
            print(f"当前位置: {pos:.2f}°")
        else:
            print("当前位置: 读取失败")
        
        # 8. 回零
        motor.trigger_home(HomeMode.NEAREST)
        
        # 9. 停止
        motor.stop()
        
        # 10. 去使能
        motor.enable(False)
        
    finally:
        motor.close()