# UK-DALE Accurate Channel Mapping
# 基于官方labels.dat文件的准确通道映射

## Building 1 (House 1)

| Appliance | Channel | Official Name |
|-----------|---------|---------------|
| **Fridge** | 12 | fridge |
| **Washing Machine** | 5 | washing_machine |
| **Dishwasher** | 6 | dishwasher |
| **Kettle** | 10 | kettle |
| **Microwave** | 13 | microwave |
| **Mains** | 1 | aggregate |

## Building 2 (House 2) - 当前使用

| Appliance | Channel | Official Name |
|-----------|---------|---------------|
| **Fridge** | 14 | fridge |
| **Washing Machine** | 12 | washing_machine |
| **Dishwasher** | 13 | dish_washer |
| **Kettle** | 8 | kettle |
| **Microwave** | 15 | microwave |
| **Mains** | 1 | aggregate |

## 如何查看Channel名称

### 方法1: 直接查看labels.dat文件

```bash
# Building 1
cat NILM-main/dataset_preprocess/UK_DALE/house_1/labels.dat

# Building 2
cat NILM-main/dataset_preprocess/UK_DALE/house_2/labels.dat
```

### 方法2: 使用Python查看

```python
import pandas as pd

# Building 1
labels1 = pd.read_csv('NILM-main/dataset_preprocess/UK_DALE/house_1/labels.dat', 
                      sep=' ', header=None, names=['channel', 'name'])
print("Building 1 Channels:")
print(labels1.to_string(index=False))

# Building 2
labels2 = pd.read_csv('NILM-main/dataset_preprocess/UK_DALE/house_2/labels.dat', 
                      sep=' ', header=None, names=['channel', 'name'])
print("\nBuilding 2 Channels:")
print(labels2.to_string(index=False))
```

### 方法3: 查找特定电器的Channel

```python
import pandas as pd

def find_appliance_channel(house, appliance_name):
    """查找特定电器的channel号"""
    labels_file = f'NILM-main/dataset_preprocess/UK_DALE/house_{house}/labels.dat'
    labels = pd.read_csv(labels_file, sep=' ', header=None, names=['channel', 'name'])
    
    # 搜索包含appliance_name的行
    result = labels[labels['name'].str.contains(appliance_name, case=False)]
    
    if len(result) > 0:
        print(f"House {house} - {appliance_name}:")
        print(result.to_string(index=False))
    else:
        print(f"House {house} - {appliance_name}: Not found")
    
    return result

# 示例：查找所有buildings中的fridge
find_appliance_channel(1, 'fridge')
find_appliance_channel(2, 'fridge')
```

### 方法4: 一键查看所有目标电器

```python
import pandas as pd

target_appliances = ['fridge', 'washing_machine', 'dishwasher', 'kettle', 'microwave']

for house in [1, 2]:
    print(f"\n{'='*60}")
    print(f"Building {house}")
    print('='*60)
    
    labels_file = f'NILM-main/dataset_preprocess/UK_DALE/house_{house}/labels.dat'
    labels = pd.read_csv(labels_file, sep=' ', header=None, names=['channel', 'name'])
    
    for appliance in target_appliances:
        result = labels[labels['name'].str.contains(appliance, case=False)]
        if len(result) > 0:
            for _, row in result.iterrows():
                print(f"  {appliance:20s} -> Channel {row['channel']:2d} ({row['name']})")
```

---

## 推荐配置 - 使用两个Building

```python
params_appliance = {
    'kettle': {
        'windowlength': 599,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [10, 8],  # Building 1: 10, Building 2: 8
        'train_build': [1, 2],
        'test_build': 2,
    },
    'microwave': {
        'windowlength': 599,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [1, 2],
        'channels': [13, 15],  # Building 1: 13, Building 2: 15
        'train_build': [1, 2],
        'test_build': 2,
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2],
        'channels': [12, 14],  # Building 1: 12, Building 2: 14
        'train_build': [1, 2],
        'test_build': 2,
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2],
        'channels': [6, 13],  # Building 1: 6, Building 2: 13
        'train_build': [1, 2],
        'test_build': 2,
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2],
        'channels': [5, 12],  # Building 1: 5, Building 2: 12
        'train_build': [1, 2],
        'test_build': 2,
    }
}
```

## 完整的Building 1 Channel列表

```
Channel 1:  aggregate
Channel 2:  boiler
Channel 3:  solar_thermal_pump
Channel 4:  laptop
Channel 5:  washing_machine ✓
Channel 6:  dishwasher ✓
Channel 7:  tv
Channel 8:  kitchen_lights
Channel 9:  htpc
Channel 10: kettle ✓
Channel 11: toaster
Channel 12: fridge ✓
Channel 13: microwave ✓
Channel 14: lcd_office
Channel 15: hifi_office
... (更多channels)
```

## 完整的Building 2 Channel列表

```
Channel 1:  aggregate
Channel 2:  laptop
Channel 3:  monitor
Channel 4:  speakers
Channel 5:  server
Channel 6:  router
Channel 7:  server_hdd
Channel 8:  kettle ✓
Channel 9:  rice_cooker
Channel 10: running_machine
Channel 11: laptop2
Channel 12: washing_machine ✓
Channel 13: dish_washer ✓
Channel 14: fridge ✓
Channel 15: microwave ✓
Channel 16: toaster
Channel 17: playstation
Channel 18: modem
Channel 19: cooker
```

## 使用方法

### 更新预处理脚本

在 `multivariate_ukdale_preprocess.py` 中更新 `params_appliance`:

```python
params_appliance = {
    'fridge': {
        'houses': [1, 2],
        'channels': [12, 14],  # 更新为正确的channels
        ...
    },
    'washingmachine': {
        'houses': [1, 2],
        'channels': [5, 12],  # 更新为正确的channels
        ...
    },
    # ... 其他电器
}
```

### 运行预处理

```bash
# 使用两个buildings获取更多训练数据
python preprocessing/multivariate_ukdale_preprocess.py \
  --appliance_name fridge \
  --data_dir "NILM-main/dataset_preprocess/UK_DALE/"
```

这样会从Building 1和Building 2提取数据，增加训练数据量！
