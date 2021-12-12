from dataset.mapillary_sls.MSLS import MSLS

if __name__ == '__main__':
    train_dataset = MSLS('/home/taowenyin/MyCode/Dataset/Mapillary_Street_Level_Sequences', mode='test',
                         cities_list='trondheim', batch_size=5)

    print(len(train_dataset.db_images_key))