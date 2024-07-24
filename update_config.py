import os
import json

def traverse_directory(base_path):
    """遍历目录，返回目录结构"""
    dir_structure = {"vision": {}, "language": {}}
    
    for category in dir_structure.keys():
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue
        
        for app in os.listdir(category_path):
            app_path = os.path.join(category_path, app)
            if not os.path.isdir(app_path):
                continue
            
            if app not in dir_structure[category]:
                dir_structure[category][app] = {}
                
            for model in os.listdir(app_path):
                model_path = os.path.join(app_path, model)
                # model_file = os.path.join(model_path, model + '.py')
                # if os.path.isfile(model_file):
                if os.path.isdir(model_path):
                    dir_structure[category][app][model] = {}
    
    return dir_structure

def convert_to_config_format(dir_structure):
    """将目录结构转换为配置文件格式"""
    config = {
        "categories": {},
        "applications": {},
        "models": {}
    }
    
    sorted_categories = sorted(dir_structure.keys())

    for i, category in enumerate(sorted_categories, start=1):
        config["categories"][str(i)] = category
        config["applications"][category] = {}
        config["models"][category] = {}

        sorted_applications = sorted(dir_structure[category].keys())
        
        for j, application in enumerate(sorted_applications, start=1):
            config["applications"][category][str(j)] = application
            config["models"][category][application] = {}

            sorted_models = sorted(dir_structure[category][application].keys())
            
            for k, model in enumerate(sorted_models, start=1):
                config["models"][category][application][str(k)] = model

    return config

def update_config_file(config_file_path, base_path):
    """更新配置文件"""
    dir_structure = traverse_directory(base_path)
    config = convert_to_config_format(dir_structure)
    
    with open(config_file_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Python 文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目目录的绝对路径（根据实际情况调整）
    project_base_path = script_dir
    # 配置文件的路径
    config_file_path = os.path.join(script_dir, 'config.json')

    update_config_file(config_file_path, project_base_path)
