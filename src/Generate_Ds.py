from CU_Dataset_Factory import CU_Dataset_Factory
from CU_Dataset_Factory import Hf_Loader, Local_Loader
"""
    Script use to generate argumented datasets used on Colab env.
"""
if __name__ == '__main__':
    print('Cultural Dataset argumentation start')
    factory = CU_Dataset_Factory('.')
    train_l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", 'train')
    validation_l = Hf_Loader("sapienzanlp/nlp2025_hw1_cultural_dataset", 'validation')
    test_l = Local_Loader('test_unlabeled.csv')

    # dataset generation for ML-Model
    factory.produce(train_l, 'train.tsv', ['languages', 'num_langs', 'reference', 'n_mod', 'back_links', 'n_visits','G','category', 'subcategory', 'type'], 'label', 49)
    factory.produce(validation_l, 'validation.tsv', ['languages', 'num_langs', 'reference', 'n_mod', 'back_links', 'n_visits','G','category', 'subcategory', 'type'], 'label', 49)
    factory.produce(test_l, "test.csv", ['languages', 'num_langs', 'reference', 'n_mod', 'back_links', 'n_visits','G','category', 'subcategory', 'type'], None, 45)
    
    # Dataset generation for Transformers-Model
    factory.produce(train_l, 'd_tr_train.tsv', ['description'], 'label', 45)
    factory.produce(validation_l, 'd_tr_validation.tsv', ['description'], 'label', 45)
    factory.produce(test_l, 'd_tr_test.tsv', ['description'], None, 45)

    factory.produce(train_l, 'i_tr_train.csv', ['intro'], 'label', 45)
    factory.produce(validation_l, 'i_tr_validation.csv', ['intro'], 'label', 45)
    factory.produce(test_l, 'i_tr_test.csv', ['intro'], None, 45)

    factory.produce(train_l, 'f_tr_train.csv', ['full_page'], 'label', 45)
    factory.produce(validation_l, 'f_tr_validation.csv', ['full_page'], 'label', 45)
    factory.produce(test_l, 'f_tr_test.csv', ['full_page'], None, 45)
    print('End process')
else:
    print('Only script generator, please use like main script')