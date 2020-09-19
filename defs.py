import time
import numpy as np
import pandas as pd
import requests
import wrds
from bs4 import BeautifulSoup
import random
from multiprocessing import Process, Manager
import datetime
import calendar
import re


#########################################################################################################

def find_filing_urls(filing_urls, cik, filing_type):
    aux = []
    endpoint = r"https://www.sec.gov/cgi-bin/browse-edgar"
    param_dict = {'action': 'getcompany',
                  'CIK': f'{cik}',
                  'type': f'{filing_type}',
                  'owner': 'exclude',
                  'output': 'atom'}

    response = requests.get(url=endpoint, params=param_dict)
    soup = BeautifulSoup(response.content, 'lxml')
    entries = soup.find_all('entry')

    for entry in entries:
        if entry.find('filing-type').text == str(filing_type):
            aux.append(entry.find('link')['href'])

    filing_urls[f'{cik}'] = aux


#########################################################################################################

def find_document_urls(document_urls, filing_urls, cik):
    aux = []

    for url in filing_urls[cik]:

        df = pd.read_html(url)[0]
        name = df[df['Type'].str.contains('10-K').fillna(False)]
        name = name['Document'].str.split(' ')

        ## targetting the complete submission .txt file
        if not name.fillna(False).any():
            name = df[df['Description'].str.contains('Complete').fillna(False)]
            name = name['Document'].str.split(' ')
            name = name.values[0][0]

        else:
            name.reset_index(inplace=True, drop=True)
            name = name[0][0]

        document_url = url.rsplit('/', 1)[0] + '/' + name
        aux.append(document_url)

    document_urls[f'{cik}'] = aux

    return


#########################################################################################################

def find_raw_statements(raw_statements, document_urls, cik):
    aux = []

    signal1 = 'total assets|total current assets'
    signal2 = 'total liabilities|total current liabilities'
    signal3 = 'cash'
    signal4 = 'shareholders|stockholders|equity'  # Add 'common stock', 'partners'?

    for url in document_urls[cik]:

        dfs = pd.read_html(url)

        for statement in dfs:

            try:
                c1 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal1,
                                                                                                       flags=re.IGNORECASE,
                                                                                                       na=False).any() or statement.iloc[
                                                                                                                          :,
                                                                                                                          1].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal1, flags=re.IGNORECASE,
                                                                          na=False).any() or statement.iloc[:,
                                                                                             2].astype(str).map(
                    lambda x: x.replace('  ', ' ')).str.contains(signal1, flags=re.IGNORECASE,
                                                                 na=False).any() or statement.iloc[:, 3].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal1, flags=re.IGNORECASE, na=False).any()

                c2 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal2,
                                                                                                       flags=re.IGNORECASE,
                                                                                                       na=False).any() or statement.iloc[
                                                                                                                          :,
                                                                                                                          1].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal2, flags=re.IGNORECASE,
                                                                          na=False).any() or statement.iloc[:,
                                                                                             2].astype(str).map(
                    lambda x: x.replace('  ', ' ')).str.contains(signal2, flags=re.IGNORECASE,
                                                                 na=False).any() or statement.iloc[:, 3].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal2, flags=re.IGNORECASE, na=False).any()

                c3 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal3,
                                                                                                       flags=re.IGNORECASE,
                                                                                                       na=False).any() or statement.iloc[
                                                                                                                          :,
                                                                                                                          1].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal3, flags=re.IGNORECASE,
                                                                          na=False).any() or statement.iloc[:,
                                                                                             2].astype(str).map(
                    lambda x: x.replace('  ', ' ')).str.contains(signal3, flags=re.IGNORECASE,
                                                                 na=False).any() or statement.iloc[:, 3].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal3, flags=re.IGNORECASE, na=False).any()

                c4 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal4,
                                                                                                       flags=re.IGNORECASE,
                                                                                                       na=False).any() or statement.iloc[
                                                                                                                          :,
                                                                                                                          1].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal4, flags=re.IGNORECASE,
                                                                          na=False).any() or statement.iloc[:,
                                                                                             2].astype(str).map(
                    lambda x: x.replace('  ', ' ')).str.contains(signal4, flags=re.IGNORECASE,
                                                                 na=False).any() or statement.iloc[:, 3].astype(
                    str).map(lambda x: x.replace('  ', ' ')).str.contains(signal4, flags=re.IGNORECASE, na=False).any()

            except (KeyError, IndexError):
                continue

            if (c1 and c2) and (c3 and c4):
                aux.append(statement)

        if aux == []:
            print("Building the BS: f'{url}'")
            aux = build_bs(dfs)

        if aux[0] is [] and aux[1] is []:
            print("BS not constructable: f'{url}'")
            continue  # Implement alternative search to build your balance sheet

    raw_statements[f'{cik}'] = aux

    return


#########################################################################################################

def format_statements(raw_statements):
    clean_statements = {}

    for cik in raw_statements:

        aux = []

        for statement in raw_statements[cik]:
            # Maybe check first, if type is a list (constructable BS)
            clean_statement = statement.T.reset_index(drop=False).astype(str).T
            clean_statement = clean_statement.loc[:,
                              clean_statement.isin(['$', np.nan, 'nan', '-', '—', '\u200b']).mean() < .5]
            clean_statement = clean_statement.loc[:, clean_statement.isnull().mean() < .55]
            clean_statement = clean_statement.dropna(axis=0, how='all')
            columns_to_drop = []
            for i in range(len(clean_statement.columns) - 1):
                if clean_statement[clean_statement.columns[i]].eq(
                        clean_statement[clean_statement.columns[i + 1]]).mean() > .7:
                    columns_to_drop.append(clean_statement.columns[i])
            clean_statement.drop(columns=columns_to_drop, inplace=True)
            clean_statement.reset_index(inplace=True, drop=True)

            year = datetime.datetime.today().year
            r = [str(x) for x in list(range(year, year - 30, -1))]

            copy = statement.T.reset_index(drop=False).T.astype(str).copy()

            copy = copy.applymap(lambda x: formatter(x, 'year'))
            years = ['NA']
            dates = ['NA']
            years_masked = copy[copy.isin(r)].dropna(axis=0, how='all').dropna(axis=1, how='all')

            if not years_masked.empty:
                years = years_masked.iloc[0].unique().tolist()
                if len(years) == 1:
                    years = years * (len(clean_statement.columns) - 1)

            if type(years_masked) == pd.core.frame.DataFrame:
                if years_masked.empty:
                    years = ['NA'] * (len(clean_statement.columns) - 1)
                    print('Year missing')

            if type(years_masked) == list:
                if years_masked == []:
                    years = ['NA'] * (len(clean_statement.columns) - 1)
                    print('Year missing')

            copy = statement.T.reset_index(drop=False).T.astype(str).copy()
            months = calendar.month_name[1:13]

            mask = copy.iloc[:, 1].str.contains('|'.join(months))
            for column in copy.iloc[:, 2:]:
                add = copy[column].str.contains('|'.join(months))
                mask = pd.concat([mask, add], axis=1)

            dates_masked = copy[mask].dropna(axis=0, how='all').dropna(axis=1, how='all')

            if not dates_masked.empty:
                dates = dates_masked.iloc[0].unique().tolist()
                dates = list(set([x.replace('\xa0', ' ').split(',')[0] for x in dates if type(x) == str]))
                dates.reverse()

            if type(dates_masked) == pd.core.frame.DataFrame:
                if dates_masked.empty:
                    dates = ['NA']
                    print('Date missing')

            if type(dates_masked) == list:
                if dates_masked == []:
                    dates = ['NA']
                    print('Date missing')

            if len(dates) == 1:
                dates = dates * len(years)

            column_list = [f'{x}, {y}' for x, y in zip(dates, years)]

            column_list = ['Financial position'] + column_list

            try:
                clean_statement.columns = column_list
            except (ValueError):
                print('Dimension mismatch: Column list & Frame')
                continue

            clean_statement.reset_index(inplace=True, drop=True)

            if clean_statement.columns.duplicated().any():
                clean_statement = clean_statement.loc[:, [True] + list(clean_statement.columns.duplicated())[1:]]

            try:  # for income statement --> find unique signal
                clean_statement[column_list[0]] = clean_statement[column_list[0]].map(lambda x: x.replace('\x92', "'"))
                clean_statement[column_list[0]] = clean_statement[column_list[0]].map(lambda x: x.replace('\xa0', " "))
                clean_statement[column_list[0]] = clean_statement[column_list[0]].map(lambda x: x.replace('’', "'"))

            except (ValueError):
                continue

            if all(x == clean_statement.columns[1] for x in clean_statement.columns[1:]):
                continue

            try:
                for i in clean_statement.columns[1:]:
                    clean_statement[i] = clean_statement[i].map(lambda x: x.replace('(', '-'))
                    clean_statement[i] = clean_statement[i].map(lambda x: x.replace(',', ''))
                    clean_statement[i] = clean_statement[i].map(lambda x: x.replace('—', '0'))
                    clean_statement[i] = clean_statement[i].map(lambda x: x.replace('\x96', '0'))
                    clean_statement[i] = clean_statement[i].map(lambda x: x.replace('\x97', '0'))

                    clean_statement[i] = pd.to_numeric(clean_statement[i], errors='coerce')

            except (AttributeError):
                continue

            for column in statement.columns:
                if statement[column].astype(str).str.contains('million', flags=re.IGNORECASE, na=False).any():
                    clean_statement['Binary: Numbers in millions'] = 1
            else:
                clean_statement['Binary: Numbers in millions'] = pd.NA

            aux.append(clean_statement)

        clean_statements[cik] = aux

    return clean_statements


#########################################################################################################

def create_variable_dataframe(clean_statements, var_dict):
    variable_dataframe = pd.DataFrame(columns=(['date', 'cik'] + [x for x in var_dict] + ['In_M_Binary']))

    for cik in clean_statements:

        frame = pd.DataFrame(columns=(['date', 'cik'] + [x for x in var_dict] + ['In_M_Binary']))

        for statement in clean_statements[cik]:

            dates = statement.columns[1:3].tolist()
            binary = statement['Binary: Numbers in millions'].values.tolist()[0]
            df = pd.DataFrame(columns=(['date', 'cik'] + [x for x in var_dict] + ['In_M_Binary']))
            df['date'] = dates
            df['In_M_Binary'] = binary
            df['cik'] = cik

            for var in var_dict:
                # Following formatting should be moved to previous function
                row = statement[statement.iloc[:, 0].map(lambda x: x.lower().replace('  ', ' ')).str.contains(
                    '|'.join(var_dict[var]))]
                try:
                    data = row[dates].values.tolist()[0]
                except (IndexError):
                    data = [pd.NA] * 2
                df[f'{var}'] = data

            frame = pd.concat(([frame, df]))

        frame = frame[~ frame['date'].duplicated()].sort_values(by='date', ascending=False)

        variable_dataframe = pd.concat([variable_dataframe, frame])

    variable_dataframe = variable_dataframe[~ variable_dataframe.index.duplicated()]
    variable_dataframe = variable_dataframe.sort_index(ascending=False)

    return variable_dataframe


########################################################################################

def concat_variable_dataframes(dataframe):
    pd.concat([dataframe])
    pass


########################################################################################

def add_gvkey(df):
    if isinstance(df, list):
        df = pd.DataFrame({'ciks': df})

    if (not isinstance(df, pd.DataFrame)) or (not 'ciks' in df.columns.values.tolist()):
        print('''Argument must be dataframe (containing 'ciks') or list of CIKs''')
        return

    db = wrds.Connection(wrds_username='gvncore')
    table = db.get_table(library='crsp_a_ccm', table='ccm_lookup')
    gvkeys = []

    for cik in df['ciks']:
        masked_table = table[table['cik'] == cik]
        gvkey = masked_table.iloc[0].loc['gvkey']
        gvkeys.append(gvkey)

    gvkeys_df = pd.DataFrame({'gvkeys': gvkeys})

    augmented_df = pd.merge(gvkeys_df, df, how='outer', left_index=True, right_index=True)

    return augmented_df


########################################################################################

def create_cik_sample(sample_size, table):
    cik_sample = []
    while len(cik_sample) < sample_size:
        n = random.randint(0, len(table) - 1)
        cik = table.iloc[n]['cik']
        if (cik in cik_sample) or (cik == None):
            continue
        else:
            cik_sample.append(cik)

    return cik_sample


########################################################################################

def formatter(x, type):
    if type == 'year':
        if x is not np.nan:
            if isinstance(x, str) and ',' in x:
                x = x.split(',')[1].strip()

            x = x.split('(')[0]
            x = x.split('.')[0]

        return x

    if type == 'date':
        if x is not np.nan:
            if isinstance(x, str) and ',' in x:
                x = x.split(',')[0]

        return x


########################################################################################

def del_txts(document_urls):
    for cik in document_urls:
        document_urls[cik] = [x for x in document_urls[cik] if x.endswith('.htm')]

    return document_urls


########################################################################################

def del_emptys(filing_urls):
    aux = [cik for cik in filing_urls if filing_urls[cik] == []]

    for cik in aux:
        del filing_urls[cik]

    return filing_urls


########################################################################################

def collect_document_urls(sample_size):
    db = wrds.Connection(wrds_username='gvncore')
    table = db.get_table(library='comp', table='funda', columns=['cik'])

    docs = {}

    while len(docs) < sample_size:
        cik_sample = create_cik_sample(sample_size, table)
        filing_urls = m.multiprocessor_filings(cik_sample)
        filing_urls = del_emptys(filing_urls)
        document_urls = m.multiprocessor_documents(filing_urls)
        document_urls = del_txts(document_urls)
        document_urls = del_emptys(document_urls)
        docs.update(document_urls)

    aux = []
    for i, cik in enumerate(docs):
        if i > sample_size - 1:
            aux.append(cik)
    for cik in aux:
        del docs[cik]

    return docs


########################################################################################

def build_bs(dfs):
    aux_left = []
    aux_right = []

    signal_left1 = 'total assets|total current assets'
    signal_left2 = 'cash'
    signal_right1 = 'total liabilities|total current liabilities'
    signal_right2 = 'shareholders|stockholders|equity'

    for statement in dfs:

        try:
            c1 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left1,
                                                                                                   flags=re.IGNORECASE,
                                                                                                   na=False).any() or statement.iloc[
                                                                                                                      :,
                                                                                                                      1].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left1, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 2].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left1, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 3].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left1, flags=re.IGNORECASE, na=False).any()

            c2 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left2,
                                                                                                   flags=re.IGNORECASE,
                                                                                                   na=False).any() or statement.iloc[
                                                                                                                      :,
                                                                                                                      1].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left2, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 2].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left2, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 3].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_left2, flags=re.IGNORECASE, na=False).any()

            c3 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right1,
                                                                                                   flags=re.IGNORECASE,
                                                                                                   na=False).any() or statement.iloc[
                                                                                                                      :,
                                                                                                                      1].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right1, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 2].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right1, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 3].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right1, flags=re.IGNORECASE,
                                                                      na=False).any()

            c4 = statement.iloc[:, 0].astype(str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right2,
                                                                                                   flags=re.IGNORECASE,
                                                                                                   na=False).any() or statement.iloc[
                                                                                                                      :,
                                                                                                                      1].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right2, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 2].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right2, flags=re.IGNORECASE,
                                                                      na=False).any() or statement.iloc[:, 3].astype(
                str).map(lambda x: x.replace('  ', ' ')).str.contains(signal_right2, flags=re.IGNORECASE,
                                                                      na=False).any()

        except (KeyError, IndexError):
            continue

        if c1 and c2:
            aux_left.append(statement)

        if c3 and c4:
            aux_right.append(statement)

    aux = [aux_left] + [aux_right]

    return aux
