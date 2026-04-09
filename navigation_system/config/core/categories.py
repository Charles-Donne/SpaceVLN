"""Static semantic category definitions used by mapping and landmark modules."""

from typing import List


class CategoryConfig:
    """Static category bundle for mapping, landmark, and detection class lists."""
    
    def __init__(self, 
                 mapping_classes: List[str],
                 landmark_classes: List[str]):
        """
        Args:
            mapping_classes: 建图基础类别（固定）
            landmark_classes: Landmark类别（动态）
        """
        # 基础类别（不变）
        self._mapping_classes = mapping_classes.copy()
        self._landmark_classes = landmark_classes.copy()
        
        self._detection_classes = self._mapping_classes + self._landmark_classes
    
    # ========== 属性访问 ==========
    
    @property
    def mapping_classes(self) -> List[str]:
        """建图基础类别（floor, stairs, wall等）"""
        return self._mapping_classes.copy()
    
    @property
    def landmark_classes(self) -> List[str]:
        """Landmark类别（bed, chair, table等）"""
        return self._landmark_classes.copy()
    
    @property
    def detection_classes(self) -> List[str]:
        """完整检测类别（mapping + landmark）"""
        return self._detection_classes.copy()
    
    # ========== 类别查询 ==========
    
    def is_mapping_class(self, class_name: str) -> bool:
        """判断是否为建图类别"""
        return class_name in self._mapping_classes
    
    def is_landmark_class(self, class_name: str) -> bool:
        """判断是否为Landmark类别"""
        return class_name in self._landmark_classes
    
    def get_class_type(self, class_name: str) -> str:
        """
        获取类别类型
        
        Returns:
            'mapping', 'landmark', 或 'unknown'
        """
        if self.is_mapping_class(class_name):
            return 'mapping'
        elif self.is_landmark_class(class_name):
            return 'landmark'
        else:
            return 'unknown'
    
    def get_statistics(self) -> dict:
        """
        获取静态类别统计信息
        
        Returns:
            统计字典
        """
        return {
            'total_mapping': len(self._mapping_classes),
            'total_landmark': len(self._landmark_classes),
            'total_detection': len(self._detection_classes),
            'mapping_classes': self._mapping_classes,
            'landmark_classes': self._landmark_classes,
        }
    
    def print_summary(self):
        """打印类别配置摘要"""
        stats = self.get_statistics()
        print(
            f"  Categories: mapping={stats['total_mapping']} "
            f"landmark={stats['total_landmark']} detection={stats['total_detection']}"
        )
    
    # ========== 复制和重置 ==========
    
    def copy(self) -> 'CategoryConfig':
        """创建副本"""
        new_config = CategoryConfig(
            self._mapping_classes,
            self._landmark_classes
        )
        return new_config


# ========== 便捷函数 ==========

def create_category_config() -> CategoryConfig:
    """
    创建默认静态类别配置（从constant.py读取）
    
    Returns:
        CategoryConfig实例
    """
    from navigation_system.config.core.constants import mapping_classes, landmark_classes
    return CategoryConfig(mapping_classes, landmark_classes)


def create_custom_category_config(
    mapping_classes: List[str],
    landmark_classes: List[str]
) -> CategoryConfig:
    """
    创建自定义类别配置
    
    Args:
        mapping_classes: 自定义建图类别
        landmark_classes: 自定义Landmark类别
    
    Returns:
        CategoryConfig实例
    """
    return CategoryConfig(mapping_classes, landmark_classes)
