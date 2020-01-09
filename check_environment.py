import importlib
import os

packages = ['numpy', 'matplotlib', 'pandas', 'xlrd', 'wordcloud', 'seaborn', 'folium', 'sklearn', 'pydotplus', 'basemap', 'nodejs', 'tensorflow', 'ipywidgets', 'jupyterlab', 'rise']

bad = []
for package in packages:
    try:
        importlib.import_module(package)
    except ImportError:
        if package == 'nodejs':
            if os.system("node --version") == 0:
                pass
        else:  
            if package == 'nodejs':
                print('nodejs is not installed')
            bad.append(f"Can't import {package}")
else:
    if len(bad) > 0:
        print('Your workshop environment is not yet fully set up:')
        print('\n'.join(bad))
    else:
        print("Your workshop environment is set up")

