import pandas as pd
df_fujiwara=pd.read_csv('/workspace/WebScraping/csv/fujiwara.csv',index_col=0,encoding='utf-8')
df_genji=pd.read_csv('/workspace/WebScraping/csv/genji.csv',index_col=0,encoding='utf-8')
df_heisi=pd.read_csv('/workspace/WebScraping/csv/heisi.csv',index_col=0,encoding='utf-8')
df_kajin=pd.read_csv('/workspace/WebScraping/csv/kajin.csv',index_col=0,encoding='utf-8')
df_syosetu=pd.read_csv('/workspace/WebScraping/csv/syosetu.csv',index_col=0,encoding='utf-8')
df_meiji=pd.read_csv('/workspace/WebScraping/csv/meiji.csv',index_col=0,encoding='utf-8')

df_list=[df_genji,df_heisi,df_kajin,df_syosetu,df_meiji]
df=df_fujiwara.copy()
for df_tmp in df_list:
    df=pd.concat([df,df_tmp],axis=0)

df.to_csv('/workspace/WebScraping/csv/ijin.csv')
df.to_csv('/workspace/ML/ijin.csv')