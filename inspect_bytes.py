from pathlib import Path
path = Path('train_model.py')
data = path.read_bytes()
# find the substring for prediction output
needle = b'st.write(f"Bashorat qilingan qiymat:'
idx = data.find(needle)
print('idx', idx)
print(data[idx-30:idx+200])
