import numpy as np
import pandas as pd
s = pd.Series([1,3,5,np.nan,6,8]) #series
print(s)

df =  pd.DataFrame(np.random.randn(6,4)) # DataFrame
print(df) 

dates = pd,date_range('20130101',periods = 6)

