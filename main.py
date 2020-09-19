import time
import defs as f
import multiprocessors as m

#########################################################################################################

document_urls = collect_document_urls(100)

#########################################################################################################

raw_statements = m.multiprocessor_statements(document_urls)
clean_statements = f.format_statements(raw_statements)

#########################################################################################################

dfs = m.multiprocessor_dfs(document_urls)
filtered_dfs = m.multiprocessor_filter(dfs, 'total assets')

#########################################################################################################

var_dict = {
    'AT': ['total assets'],
    'LB': ['total liabilities']}

variable_dataframe = create_variable_dataframe(clean_statements, var_dict)
