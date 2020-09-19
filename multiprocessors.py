import time
from multiprocessing import Process, Manager
import defs as f
import variable_scraper as s


### Multiprocessors

def multiprocessor_filings(cik_sample):
    starttime = time.time()
    with Manager() as manager:
        filing_urls = manager.dict()
        processes = []

        for cik in cik_sample:
            p = Process(target=f.find_filing_urls, args=(filing_urls, cik, '10-K',))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        filing_urls = dict(filing_urls)
    print(f'length of the list: {len(filing_urls)} | Computing time: {time.time() - starttime}')

    return filing_urls


#########################################################################################################

def multiprocessor_documents(filing_urls):
    starttime = time.time()
    with Manager() as manager:
        document_urls = manager.dict()
        processes = []

        for cik in filing_urls:
            p = Process(target=f.find_document_urls, args=(document_urls, filing_urls, cik,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        document_urls = dict(document_urls)
    print(f'length of the list: {len(document_urls)} | Computing time: {time.time() - starttime}')

    return document_urls


#########################################################################################################

def multiprocessor_statements(document_urls):
    starttime = time.time()
    with Manager() as manager:
        raw_statements = manager.dict()
        processes = []

        for cik in [x for x in document_urls.keys() if document_urls[x] != []]:
            p = Process(target=f.find_raw_statements, args=(raw_statements, document_urls, cik,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        raw_statements = dict(raw_statements)
    print(f'length of the list: {len(raw_statements)} | Computing time: {time.time() - starttime}')

    return raw_statements


#########################################################################################################

def multiprocessor_dfs(document_urls):
    starttime = time.time()
    with Manager() as manager:
        dfs = manager.dict()
        processes = []

        for cik in document_urls:
            p = Process(target=s.scrape_dfs, args=(dfs, document_urls, cik,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        dfs = dict(dfs)
    print(f'length of the list: {len(dfs)} | Computing time: {time.time() - starttime}')

    return dfs


#########################################################################################################

def multiprocessor_filter(dfs, variable):
    starttime = time.time()
    with Manager() as manager:
        filtered_dfs = manager.dict()
        processes = []

        for cik in dfs:
            p = Process(target=s.filter_dfs, args=(filtered_dfs, dfs, cik, variable,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        filtered_dfs = dict(filtered_dfs)
    print(f'length of the list: {len(filtered_dfs)} | Computing time: {time.time() - starttime}')

    return dfs
