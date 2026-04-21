"""Static semantic category definitions used by mapping and landmark modules."""

from typing import List


class CategoryConfig:
    """Static category bundle for mapping, landmark, and detection class lists."""
    
    def __init__(self, 
                 mapping_classes: List[str],
                 landmark_classes: List[str]):
        """
        Args:
            mapping_classes: Static mapping classes.
            landmark_classes: Dynamic landmark classes.
        """
        # Static mapping categories
        self._mapping_classes = mapping_classes.copy()
        self._landmark_classes = landmark_classes.copy()
        
        self._detection_classes = self._mapping_classes + self._landmark_classes
    
    # ========== Accessors ==========
    
    @property
    def mapping_classes(self) -> List[str]:
        """Static mapping categories such as floor, stairs, and wall."""
        return self._mapping_classes.copy()
    
    @property
    def landmark_classes(self) -> List[str]:
        """Landmark categories such as bed, chair, and table."""
        return self._landmark_classes.copy()
    
    @property
    def detection_classes(self) -> List[str]:
        """Complete detection categories (`mapping + landmark`)."""
        return self._detection_classes.copy()
    
    # ========== Category queries ==========
    
    def is_mapping_class(self, class_name: str) -> bool:
        """Return whether the class belongs to mapping categories."""
        return class_name in self._mapping_classes
    
    def is_landmark_class(self, class_name: str) -> bool:
        """Return whether the class belongs to landmark categories."""
        return class_name in self._landmark_classes
    
    def get_class_type(self, class_name: str) -> str:
        """
        Return the category family.

        Returns:
            `mapping`, `landmark`, or `unknown`.
        """
        if self.is_mapping_class(class_name):
            return 'mapping'
        elif self.is_landmark_class(class_name):
            return 'landmark'
        else:
            return 'unknown'
    
    def get_statistics(self) -> dict:
        """
        Return summary statistics for the configured categories.
        """
        return {
            'total_mapping': len(self._mapping_classes),
            'total_landmark': len(self._landmark_classes),
            'total_detection': len(self._detection_classes),
            'mapping_classes': self._mapping_classes,
            'landmark_classes': self._landmark_classes,
        }
    
    def print_summary(self):
        """Print a category configuration summary."""
        stats = self.get_statistics()
        print(
            f"  Categories: mapping={stats['total_mapping']} "
            f"landmark={stats['total_landmark']} detection={stats['total_detection']}"
        )
    
    # ========== Copy helpers ==========
    
    def copy(self) -> 'CategoryConfig':
        """Create a shallow copy."""
        new_config = CategoryConfig(
            self._mapping_classes,
            self._landmark_classes
        )
        return new_config


# ========== Convenience helpers ==========

def create_category_config() -> CategoryConfig:
    """
    Create the default category config from `constants.py`.
    """
    from navigation_system.config.core.constants import mapping_classes, landmark_classes
    return CategoryConfig(mapping_classes, landmark_classes)


def create_custom_category_config(
    mapping_classes: List[str],
    landmark_classes: List[str]
) -> CategoryConfig:
    """
    Create a custom category config.

    Args:
        mapping_classes: Custom mapping classes.
        landmark_classes: Custom landmark classes.
    """
    return CategoryConfig(mapping_classes, landmark_classes)
