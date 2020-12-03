import pandas as pd



SumFolder = "E:\\QPD Speed Data Collection\\"
SumFileName="QPD-Speed Summary.xlsx"
SumFile = pd.read_excel(SumFolder+SumFileName)
#SumFile=SumFile.set_index("Samples",drop=False)
print("hello")