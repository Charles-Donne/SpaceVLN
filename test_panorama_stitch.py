#!/usr/bin/env python3
"""
全景图拼接测试程序
使用OpenCV Stitcher类测试3张图片的拼接效果

使用方法:
python test_panorama_stitch.py <图片1路径> <图片2路径> <图片3路径>

或者直接运行测试默认路径:
python test_panorama_stitch.py
"""

import cv2
import sys
import os
import numpy as np


def stitch_panorama(image_paths, mode='panorama'):
    """
    使用OpenCV Stitcher拼接全景图
    
    Args:
        image_paths: 图片路径列表，按顺序（如左-中-右）
        mode: 'panorama' 或 'scans'
              - panorama: 适合水平旋转拍摄的全景图（推荐）
              - scans: 适合扫描文档等平面拼接
    
    Returns:
        status: 拼接状态码
            0 (OK): 成功
            1 (ERR_NEED_MORE_IMGS): 需要更多图片
            2 (ERR_HOMOGRAPHY_EST_FAIL): 单应性估计失败
            3 (ERR_CAMERA_PARAMS_ADJUST_FAIL): 相机参数调整失败
        panorama: 拼接后的全景图（如果成功）
    """
    print(f"\n{'='*60}")
    print(f"OpenCV Stitcher 全景拼接测试")
    print(f"{'='*60}\n")
    
    # 读取图片
    images = []
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"❌ 错误: 图片不存在 - {path}")
            return None, None
        
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 错误: 无法读取图片 - {path}")
            return None, None
        
        images.append(img)
        print(f"✓ 已加载图片{i+1}: {path}")
        print(f"  尺寸: {img.shape[1]}x{img.shape[0]}")
    
    print(f"\n共加载 {len(images)} 张图片\n")
    
    # 创建Stitcher对象
    if mode == 'panorama':
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        print("使用模式: PANORAMA (水平旋转全景)")
    else:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        print("使用模式: SCANS (平面扫描)")
    
    # 执行拼接
    print("\n开始拼接...")
    status, panorama = stitcher.stitch(images)
    
    # 状态码映射
    status_messages = {
        0: "✓ 拼接成功 (OK)",
        1: "❌ 需要更多图片 (ERR_NEED_MORE_IMGS)",
        2: "❌ 单应性估计失败 (ERR_HOMOGRAPHY_EST_FAIL) - 可能重叠区域太少或特征点不足",
        3: "❌ 相机参数调整失败 (ERR_CAMERA_PARAMS_ADJUST_FAIL)"
    }
    
    print(f"\n状态: {status_messages.get(status, f'未知错误 ({status})')}")
    
    if status == 0:
        print(f"✓ 原始全景图尺寸: {panorama.shape[1]}x{panorama.shape[0]}")
        
        # 裁剪黑边
        print("\n正在裁剪黑边...")
        panorama_cropped = crop_black_borders(panorama)
        print(f"✓ 裁剪后尺寸: {panorama_cropped.shape[1]}x{panorama_cropped.shape[0]}")
        
        return status, panorama_cropped
    else:
        print("\n💡 拼接失败的可能原因:")
        print("   1. 图片重叠区域不足（建议30%-50%重叠）")
        print("   2. 图片模糊或特征点太少")
        print("   3. 图片顺序错误（应按拍摄顺序：左→中→右）")
        print("   4. 相机角度变化太大")
        print("\n💡 建议尝试:")
        print("   - 检查图片顺序是否正确")
        print("   - 尝试调整图片大小 (resize)")
        print("   - 尝试 mode='scans' 模式")
    
    return status, panorama


def crop_black_borders(img, threshold=10):
    """
    裁剪全景图的黑边
    
    Args:
        img: 输入图像
        threshold: 黑色阈值（0-255），小于此值视为黑色
        
    Returns:
        裁剪后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 找到非黑色区域
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # 获取最大轮廓的边界框
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 裁剪
    cropped = img[y:y+h, x:x+w]
    
    return cropped


def test_with_sample_images():
    """使用示例图片路径测试"""
    print("请提供3张图片路径，按拍摄顺序（如：左-中-右）")
    print("\n示例:")
    print("  图片1: records/episode_XXX/rgb/step-1.jpg")
    print("  图片2: records/episode_XXX/rgb/step-12.jpg")
    print("  图片3: records/episode_XXX/rgb/step-11.jpg")
    print("\n或者直接拖拽3张图片到终端")
    
    image_paths = []
    for i in range(3):
        path = input(f"\n图片{i+1}路径: ").strip()
        # 移除可能的引号
        path = path.strip('"').strip("'")
        image_paths.append(path)
    
    return image_paths


def save_comparison(images, panorama, output_path='panorama_result.jpg'):
    """保存拼接结果和对比图"""
    # 保存全景图
    cv2.imwrite(output_path, panorama)
    print(f"\n✓ 全景图已保存: {output_path}")
    
    # 创建对比图（原图 vs 全景图）
    # 缩放原图到相同高度
    h = 200  # 统一高度
    resized_originals = []
    for img in images:
        ratio = h / img.shape[0]
        w = int(img.shape[1] * ratio)
        resized = cv2.resize(img, (w, h))
        resized_originals.append(resized)
    
    # 水平拼接原图
    originals_concat = np.hstack(resized_originals)
    
    # 缩放全景图到相同高度
    ratio = h / panorama.shape[0]
    w_pano = int(panorama.shape[1] * ratio)
    panorama_resized = cv2.resize(panorama, (w_pano, h))
    
    # 垂直拼接（上：原图，下：全景图）
    # 调整宽度使其一致
    max_w = max(originals_concat.shape[1], panorama_resized.shape[1])
    
    if originals_concat.shape[1] < max_w:
        pad = np.zeros((h, max_w - originals_concat.shape[1], 3), dtype=np.uint8)
        originals_concat = np.hstack([originals_concat, pad])
    
    if panorama_resized.shape[1] < max_w:
        pad = np.zeros((h, max_w - panorama_resized.shape[1], 3), dtype=np.uint8)
        panorama_resized = np.hstack([panorama_resized, pad])
    
    comparison = np.vstack([originals_concat, panorama_resized])
    
    comparison_path = output_path.replace('.jpg', '_comparison.jpg')
    cv2.imwrite(comparison_path, comparison)
    print(f"✓ 对比图已保存: {comparison_path}")


def main():
    print("\n" + "="*60)
    print("OpenCV Stitcher 全景拼接测试程序")
    print("="*60)
    
    # 检查命令行参数
    if len(sys.argv) == 4:
        # 从命令行获取3张图片路径
        image_paths = sys.argv[1:4]
    else:
        # 交互式输入
        image_paths = test_with_sample_images()
    
    # 执行拼接
    status, panorama = stitch_panorama(image_paths, mode='panorama')
    
    if status == 0 and panorama is not None:
        # 保存结果
        images = [cv2.imread(p) for p in image_paths]
        save_comparison(images, panorama, 'panorama_result.jpg')
        
        print("\n" + "="*60)
        print("✓ 拼接成功！")
        print("="*60)
        
        # 可选：显示结果（需要GUI环境）
        try:
            cv2.imshow('Panorama Result', panorama)
            print("\n按任意键关闭预览窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("\n(无法显示预览窗口，可能是无GUI环境)")
    else:
        print("\n" + "="*60)
        print("❌ 拼接失败")
        print("="*60)
        
        # 提供更详细的调试建议
        if status == 2:  # ERR_HOMOGRAPHY_EST_FAIL
            print("\n💡 调试建议:")
            print("   1. 尝试缩小图片尺寸:")
            print("      python -c \"import cv2; img=cv2.imread('图片.jpg'); ")
            print("      cv2.imwrite('small.jpg', cv2.resize(img, (640, 480)))\"")
            print("\n   2. 尝试使用 mode='scans':")
            print("      修改代码中 stitch_panorama(..., mode='scans')")
            print("\n   3. 检查图片内容:")
            print("      - 确保有足够的纹理和特征（避免纯色墙面）")
            print("      - 确保重叠区域有明显的共同特征")


if __name__ == '__main__':
    main()
