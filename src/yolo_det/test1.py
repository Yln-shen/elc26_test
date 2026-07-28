#!/usr/bin/env python3
"""
RKNN模型在Rock4D上的摄像头测试程序
适配 RK3576 模型
"""

import os
import sys
import cv2
import numpy as np
import time
import signal
from pathlib import Path
from rknnlite.api import RKNNLite

class Rock4DYOLODetector:
    def __init__(self, model_path, class_names=None, input_size=(640, 640)):
        """
        初始化RKNN检测器
        """
        self.model_path = model_path
        self.class_names = class_names or ['bead']
        self.input_size = input_size
        
        # 性能统计
        self.fps = 0
        self.inference_time = 0
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # 初始化RKNN
        self._init_rknn()
        
        # 颜色
        self.colors = self._generate_colors()
        
        print(f"✅ 模型加载成功")
        print(f"   输入尺寸: {self.input_size[0]}x{self.input_size[1]}")
        print(f"   类别: {self.class_names}")
    
    def _init_rknn(self):
        """初始化RKNN"""
        try:
            self.rknn = RKNNLite()
            
            # 检查模型文件
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 加载模型
            print(f"📦 加载模型: {self.model_path}")
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"加载模型失败，错误码: {ret}")
            
            # 初始化运行时
            print("🔧 初始化NPU运行时...")
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化运行时失败，错误码: {ret}")
            
            print("✅ RKNN初始化成功")
            
        except Exception as e:
            print(f"❌ RKNN初始化失败: {e}")
            raise
    
    def _generate_colors(self):
        """生成颜色"""
        colors = []
        for i in range(80):
            color = (np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255))
            colors.append(color)
        return colors
    
    def preprocess(self, image):
        """
        图像预处理 - 适配 RK3576 模型 (NCHW 格式)
        """
        # 调整大小
        h, w = self.input_size
        resized = cv2.resize(image, (w, h))
        
        # BGR转RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转成 NCHW 格式 (因为模型是 ONNX, layout: NCHW)
        # HWC -> CHW
        chw = np.transpose(rgb, (2, 0, 1))
        
        # 归一化 (根据模型要求)
        # 如果模型需要归一化到 [0,1] 或 [-1,1]
        # 根据您的模型训练时的预处理方式调整
        normalized = chw.astype(np.float32) / 255.0
        
        # 添加 batch 维度
        processed = np.expand_dims(normalized, axis=0)
        
        return processed
    
    def postprocess(self, outputs, image_shape, conf_threshold=0.5):
        """
        后处理YOLO输出
        """
        detections = []
        
        try:
            # 获取输出
            if isinstance(outputs, list):
                output = outputs[0]
            else:
                output = outputs
            
            print(f"📊 输出形状: {output.shape}")
            
            # 根据输出形状处理
            if len(output.shape) == 3:
                # [batch, num_detections, 6] 或 [batch, 6, num_detections]
                batch_size, num_detections, num_attrs = output.shape
                
                # 原始图像尺寸
                orig_h, orig_w = image_shape[:2]
                scale_x = orig_w / self.input_size[1]
                scale_y = orig_h / self.input_size[0]
                
                for det in output[0]:
                    # 假设格式: [x, y, w, h, conf, class]
                    if num_attrs >= 6:
                        x, y, w, h, conf, cls = det[:6]
                    else:
                        continue
                    
                    if conf < conf_threshold:
                        continue
                    
                    # 转换为原始图像坐标
                    x1 = int((x - w/2) * scale_x)
                    y1 = int((y - h/2) * scale_y)
                    x2 = int((x + w/2) * scale_x)
                    y2 = int((y + h/2) * scale_y)
                    
                    # 裁剪到图像范围
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': int(cls)
                    })
            else:
                # 其他输出格式 - 打印调试信息
                print(f"⚠️ 未处理的输出格式: {output.shape}")
                
        except Exception as e:
            print(f"⚠️ 后处理错误: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def detect(self, image, conf_threshold=0.5):
        """
        执行检测
        """
        start_time = time.time()
        
        # 预处理
        processed = self.preprocess(image)
        
        try:
            # 推理
            outputs = self.rknn.inference(inputs=[processed])
            
            # 计算推理时间
            inference_time = (time.time() - start_time) * 1000
            
            # 后处理
            detections = self.postprocess(outputs, image.shape, conf_threshold)
            
        except Exception as e:
            print(f"❌ 推理错误: {e}")
            import traceback
            traceback.print_exc()
            return [], 0
        
        return detections, inference_time
    
    def draw_detections(self, image, detections):
        """
        绘制检测结果
        """
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class']
            
            class_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
            color = self.colors[cls % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 标签背景
            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # 标签文字
            cv2.putText(img_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 显示性能信息
        y_offset = 30
        if self.fps > 0:
            cv2.putText(img_copy, f"FPS: {self.fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        if self.inference_time > 0:
            cv2.putText(img_copy, f"Inf: {self.inference_time:.1f}ms", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        cv2.putText(img_copy, f"Count: {len(detections)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img_copy
    
    def update_fps(self):
        """更新FPS"""
        self.frame_count += 1
        self.fps_counter += 1
        
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def run_camera(self, camera_id=0, conf_threshold=0.5, display=True):
        """
        运行摄像头检测
        """
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            return False
        
        # 获取摄像头参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📷 摄像头信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"\n🔍 控制:")
        print(f"   ESC/q - 退出")
        print(f"   s     - 保存截图")
        print(f"   p     - 暂停/继续")
        print("-" * 50)
        
        window_name = 'Rock4D YOLO Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        paused = False
        running = True
        
        while running:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 无法读取摄像头帧")
                    break
                
                # 执行检测
                detections, inference_time = self.detect(frame, conf_threshold)
                
                # 更新性能统计
                self.inference_time = inference_time
                self.update_fps()
                
                # 绘制结果
                if display:
                    annotated = self.draw_detections(frame, detections)
                    cv2.imshow(window_name, annotated)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC或q退出
                break
            elif key == ord('s'):  # 保存截图
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                if not paused:
                    cv2.imwrite(filename, annotated)
                else:
                    cv2.imwrite(filename, frame)
                print(f"💾 保存截图: {filename}")
            elif key == ord('p'):  # 暂停
                paused = not paused
                print(f"⏸️ {'暂停' if paused else '继续'}")
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 检测结束")
        return True
    
    def run_video(self, video_path, conf_threshold=0.5, display=True):
        """
        运行视频文件检测
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return False
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎬 视频信息:")
        print(f"   路径: {video_path}")
        print(f"   尺寸: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        print(f"   总帧数: {total_frames}")
        print("\n按 ESC 或 q 退出")
        print("-" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行检测
            detections, inference_time = self.detect(frame, conf_threshold)
            
            # 更新性能统计
            self.inference_time = inference_time
            self.update_fps()
            
            # 绘制结果
            if display:
                annotated = self.draw_detections(frame, detections)
                cv2.imshow('Rock4D YOLO Detection', annotated)
                
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 视频处理完成")
        return True
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, 'rknn'):
                self.rknn.release()
            print("✅ 资源清理完成")
        except:
            pass
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 50)
        print("📊 性能统计")
        print("=" * 50)
        print(f"总帧数: {self.frame_count}")
        print(f"平均FPS: {self.fps:.1f}")
        print(f"平均推理时间: {self.inference_time:.1f}ms")
        print("=" * 50)


# ===== 主程序 =====
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rock4D RKNN YOLO检测测试')
    parser.add_argument('--model', type=str, 
                       default='src/yolo_det/best_rknn_model/best-rk3576.rknn',
                       help='RKNN模型路径 (.rknn文件)')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (0=摄像头, 或视频文件路径)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['bead'],
                       help='类别名称列表')
    parser.add_argument('--input-size', type=int, nargs=2, 
                       default=[640, 640],
                       help='模型输入尺寸 [height width]')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("\n💡 请指定正确的模型路径:")
        print(f"   python test1.py --model /path/to/model.rknn --source 0")
        sys.exit(1)
    
    print(f"🔍 模型: {args.model}")
    print(f"📷 源: {args.source}")
    print(f"🎯 置信度阈值: {args.conf}")
    
    # 初始化检测器
    try:
        detector = Rock4DYOLODetector(
            model_path=args.model,
            class_names=args.classes,
            input_size=tuple(args.input_size)
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        # 判断输入源
        if args.source == '0' or args.source.isdigit():
            # 摄像头
            success = detector.run_camera(
                camera_id=int(args.source),
                conf_threshold=args.conf
            )
        else:
            # 视频文件
            success = detector.run_video(
                video_path=args.source,
                conf_threshold=args.conf
            )
        
        # 打印统计
        detector.print_stats()
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()