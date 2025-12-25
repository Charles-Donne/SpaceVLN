"""
测试ground truth reference_path加载
"""
import gzip
import json
import sys

def check_reference_path(dataset_path):
    """检查数据集中是否包含reference_path"""
    print(f"📂 检查数据集: {dataset_path}")
    
    try:
        with gzip.open(dataset_path, 'rt') as f:
            data = json.load(f)
        
        print(f"✅ 成功加载数据集")
        print(f"   Episode总数: {len(data['episodes'])}")
        
        # 检查前几个episode
        has_reference_path = 0
        for i, ep in enumerate(data['episodes'][:5]):
            ep_id = ep.get('episode_id', 'N/A')
            ref_path = ep.get('reference_path', None)
            
            if ref_path is not None:
                has_reference_path += 1
                print(f"   Episode {ep_id}: ✅ reference_path存在 ({len(ref_path)}个点)")
            else:
                print(f"   Episode {ep_id}: ❌ 缺少reference_path")
        
        print(f"\n📊 统计:")
        print(f"   前5个episodes中有{has_reference_path}个包含reference_path")
        
        if has_reference_path == 0:
            print("\n⚠️  警告: 数据集不包含reference_path字段!")
            print("   这可能是预处理后的数据集，需要原始数据集来显示ground truth")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # 默认路径
        dataset_path = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz"
    
    check_reference_path(dataset_path)
