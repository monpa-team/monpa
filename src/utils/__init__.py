def ChunkS(text):
    '''Chunk sentence function
    '''
    chunk_size = 200
    chunk_char = ["。", "！", "？", "，", "END"]
    if len(text) <= chunk_size:
        return f'{text.strip()}'
    for cchar in chunk_char:
        i = text.rfind(cchar, 0, chunk_size)
        if i == -1:
            if cchar != "END":
                continue
            return f'{text[:chunk_size].strip()}<CHUNK>{ChunkS(text[chunk_size+1:])}'
        return f'{text[:i+1].strip()}<CHUNK>{ChunkS(text[i+1:])}'

def short_sentence(text):
    '''Chunking long document to multi short sentences, base on "。", "！", "？", "，"'''
    return ChunkS(text).split("<CHUNK>")

def cut_mp(text_list, num_processor):
    '''multi process: input list or nested list, specify how many processor to be worker'''
    from multiprocessing import Pool
    import monpa
    pool = Pool(processes=num_processor)
    if any(isinstance(i, list) for i in text_list):
        result = [pool.map(monpa.cut, item) for item in text_list if item]
    else:
        result = pool.map(monpa.cut, text_list)
    pool.terminate()
    del pool
    return result

def pseg_mp(text_list, num_processor):
    '''multi process: input list or nested list, specify how many processor to be worker'''
    from multiprocessing import Pool
    import monpa
    pool = Pool(processes=num_processor)
    if any(isinstance(i, list) for i in text_list):
        result = [pool.map(monpa.pseg, item) for item in text_list if item]
    else:
        result = pool.map(monpa.pseg, text_list)
    pool.terminate()
    del pool
    return result
