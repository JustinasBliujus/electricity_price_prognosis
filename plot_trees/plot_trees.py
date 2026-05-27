import os 
import re
import graphviz
import lightgbm as lgbm
import xgboost as xgb

CURRENT_DIR = current_dir = os.path.dirname(os.path.abspath(__file__))

feature_map = {
        'hour': 'Valanda',
        'day': 'Diena',
        'month': 'Mėnuo',
        'year': 'Metai',
        'dayofweek': 'Savaitės_diena',
        'quarter': 'Ketvirtis',
        'dayofyear': 'Metų_diena',
        'weekend': 'Savaitgalis',
        'lag_1': 'Atsilikimas_1h',
        'lag_2': 'Atsilikimas_2h',
        'lag_3': 'Atsilikimas_3h',
        'lag_6': 'Atsilikimas_6h',
        'lag_12': 'Atsilikimas_12h',
        'lag_24': 'Atsilikimas_24h',
        'lag_48': 'Atsilikimas_48h',
        'lag_96': 'Atsilikimas_96h',
        'lag_168': 'Atsilikimas_168h',
        'rolling_mean_6': 'Slankusis_vidurkis_6h',
        'rolling_std_6': 'Slankusis_nuokrypis_6h',
        'rolling_mean_12': 'Slankusis_vidurkis_12h',
        'rolling_std_12': 'Slankusis_nuokrypis_12h',
        'rolling_mean_24': 'Slankusis_vidurkis_24h',
        'rolling_std_24': 'Slankusis_nuokrypis_24h',
        'rolling_mean_48': 'Slankusis_vidurkis_48h',
        'rolling_std_48': 'Slankusis_nuokrypis_48h',
        'rolling_mean_168': 'Slankusis_vidurkis_168h',
        'rolling_std_168': 'Slankusis_nuokrypis_168h',
}

def plot_lgbm(model):
    en = list(feature_map.keys())
    lt = list(feature_map.values())
    model = model.booster_  
    
    print(f"isviso {model.num_trees()}")
    
    dump = model.dump_model()
    tree_sizes = {}
    for tree_index, tree in enumerate(dump["tree_info"]):
        tree_sizes[tree_index] = tree["num_leaves"]
        smallest_tree_index = min(tree_sizes, key=tree_sizes.get)
    print(f"maz {smallest_tree_index}")

    model_string = model.model_to_string()

    for i, name in sorted(enumerate(lt), key=lambda x: -len(str(x[0]))):
        model_string = model_string.replace(f'Column_{i}', name)

    new_model = lgbm.Booster(model_str=model_string)

    graph = lgbm.create_tree_digraph(new_model, tree_index=869)
    src = graph.source
    src = src.replace('label=yes', 'label=taip')
    src = src.replace('label=no', 'label=ne')
    src = re.sub(r'leaf (\d+):', r'', src)
    
    new_graph = graphviz.Source(src)
    path = os.path.join(CURRENT_DIR,"lgbm_tree")
    new_graph.render(path, format='png', cleanup=True)
    print(path)
    
def plot_xgb(model):
    en = list(feature_map.keys())
    lt = list(feature_map.values())

    booster = model.get_booster()
    print(f"isviso {booster.num_boosted_rounds()}")
    
    tree_df = booster.trees_to_dataframe()
    tree_sizes = tree_df.groupby('Tree').size()
    smallest_tree_index = tree_sizes.idxmin()
    print(f"maz {smallest_tree_index}")
    
    booster.feature_names = lt

    graph = xgb.to_graphviz(booster, tree_idx=366)

    src = graph.source
    src = src.replace('yes', 'taip')
    src = src.replace('no, missing', 'ne')
    src = re.sub(r'leaf=', r'', src)

    new_graph = graphviz.Source(src)
    path = os.path.join(CURRENT_DIR, 'xgb_tree')
    new_graph.render(path, format='png', cleanup=True)
    print(path)