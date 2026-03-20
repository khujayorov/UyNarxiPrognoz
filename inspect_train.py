import pathlib
path = pathlib.Path('train_model.py')
text = path.read_text(encoding='utf-8')
index = text.find('st.write(f"Bashorat qilingan qiymat')
print('index', index)
print(repr(text[index:index+200]))
