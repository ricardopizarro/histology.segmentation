def split_cross_valid(slices_fn,segments_fn,train,valid,test):
    train_files =[]
    validation_files=[]
    test_files=[]
    for n,segment_fn in enumerate(segments_fn):
        slice_fn=difflib.get_close_matches(segment_fn,slices_fn)[0]
        if not slice_fn:
            print("Could not find an equivalent segment file {}".format(segment_fn))
            continue
        slice_nb = os.path.basename(segment_fn)[:4]
        if slice_nb in train:
            train_files.append((slice_fn,segment_fn))
        elif slice_nb in valid:
            validation_files.append((slice_fn,segment_fn))
        elif slice_nb in test:
            test_files.append((slice_fn,segment_fn))
        else:
            print('{} is not in any subset!'.format(segment_fn))
    return train_files,validation_files,test_files


