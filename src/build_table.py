# -*- coding: utf-8 -*-
"""
Automatically building latex tables
Taken from: <other repo under anonymous submission>
"""

TEMPLATE = """
\\begin{table*}[<position>]
    \\centering
    \\caption{<caption>}
    \\label{<label>}
    \\resizebox{<resize_col>\\columnwidth}{!}{
        \\setlength{\\tabcolsep}{8pt}
        \\renewcommand{\\arraystretch}{1.2}
        \\begin{tabular}{<alignment>}
            \\toprule
            <columns> \\\\
            <midrule_1>
            <sub_columns> 
            <midrule_2>
            <data> \\\\
            \\bottomrule
        \end{tabular}
    }
\end{table*}
"""


def check_alignment_data(columns: list[str], label: str, alignment: str, data: list[list[str]]):
    if len(columns) != len(alignment):
        raise ValueError(f"Params `{label}` and `alignment` should have the same length")
    if any(len(x) != len(columns) for x in data):
        raise ValueError("Each list in `data` must have the same length as `{label}`")


def check_args(columns: list[str], alignment: str,
               data: list[list[str]], sub_columns: list[str], multicol: list[int]):
    """ Checking if format of input param will match table """
    if len(sub_columns) == 0 ^ len(multicol) == 0:
        raise ValueError("Params `sub_columns` and `multicol` must be either both True or False")
    
    if sub_columns:
        check_alignment_data(columns=sub_columns, label="sub_columns", alignment=alignment, data=data)

        if len(multicol) != len(columns):
            raise ValueError("Params `multicol` and `columns` must have the same length")
        if sum(multicol) != len(sub_columns):
            raise ValueError("The sum of integers in `multicol` must be equal to the length of `sub_columns`")
    else:   # Only main columns
        check_alignment_data(columns=columns, label="sub_columns", alignment=alignment, data=data)
    return
        

def get_start_end_multicol(multicol: list[int]) -> list[str]:
    curr_start, start, end = 1, [], []
    for x in multicol:
        start.append(curr_start)
        end.append(curr_start + x - 1)
        curr_start = end[-1] + 1
    return [f"{val}-{end[i]}" for i, val in enumerate(start)]


def build_table(columns: list[str], alignment: str,
                caption: str, label: str, position: str,
                data: list[list[str]], sub_columns: list[str] = [], multicol: list[int] = [],
                resize_col: int = 1) -> str:
    """ 
    - `data`: list of list. Each element in the list corresponds to the set of values
    for one row of the table
    """
    check_args(columns=columns, alignment=alignment, data=data, sub_columns=sub_columns, multicol=multicol)
    if sub_columns: 
        columns = ["\\multicolumn{" + str(multicol[i]) + "}{c}{" + col + "}" for i, col in enumerate(columns)]

        start_end = get_start_end_multicol(multicol=multicol)
        midrule_1 = "\n\t".join(["\\cmidrule(lr){" + x + "}" for x in start_end])
        midrule_2 = midrule_1

        sub_columns = " & ".join(sub_columns) + "\\\\"

    else:
        midrule_1 = "\\midrule"
        midrule_2 = ""
        sub_columns = ""

    columns = " & ".join(columns)
    data = "\\\ \n".join([" & ".join(['{:,}'.format(y) if not isinstance(y, str) else y for y in x]) for x in data])

    table = TEMPLATE.replace("<position>", position).replace("<caption>", caption) \
        .replace("<label>", label).replace("<alignment>", alignment) \
            .replace("<columns>", columns).replace("<midrule_1>", midrule_1) \
            .replace("<sub_columns>", sub_columns).replace("<midrule_2>", midrule_2) \
            .replace("<data>", data).replace("<resize_col>", str(resize_col))
    return table
