from CU_Dataset_Factory import CU_Dataset_Factory
from CU_Dataset_Factory import Hf_Loader

if __name__ == '__main__':
    print('Cultural Dataset argumentation start')
    factory = CU_Dataset_Factory('.')
    train_l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", 'train')
    validation_l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", 'validation')

    factory.produce(train_l, 'train.tsv', ['category', 'subcategory','type','languages','reference','num_langs', 'G', 'n_mod', 'back_links'], 'label', 10, False)
    factory.produce(validation_l, 'validation.tsv', ['category', 'subcategory','type','languages','reference','num_langs', 'G', 'n_mod', 'back_links'], 'label', 10, False)
    print('End process')
    exit(0)

print('Only script generator, please use like main script')