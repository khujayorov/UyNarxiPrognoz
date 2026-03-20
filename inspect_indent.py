from pathlib import Path
path=Path('train_model.py')
text=path.read_bytes()
needle=b'st.write(f"Bashorat qilingan qiymat:'
idx=text.find(needle)
print('idx', idx)
print(text[idx-50:idx+len(needle)+10])
