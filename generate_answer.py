import pandas
from pandas import DataFrame

df1 = pandas.read_csv("/mnt/sdc1/korLM/by_me/out/test.gender_bias.no_label.csv")
df2 = pandas.read_csv('/mnt/sdc1/korLM/by_me/out/by_me_9_bias_label.csv')
li=[df1,df2]

frame = pandas.concat(li, axis=1)

csv_write_path='/mnt/sdc1/korLM/by_me/out/by_me_9_bias.csv'
frame.to_csv(csv_write_path, sep=',',columns=['comments','label'], index=False, encoding = 'utf-8')