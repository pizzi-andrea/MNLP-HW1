
from CU_Dataset_Factory import CU_Dataset_Factory
from pathlib import PosixPath


builder = CU_Dataset_Factory(True)


validation = builder.produce(PosixPath('./dataset/'), True, False)


