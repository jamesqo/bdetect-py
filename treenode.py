def compare(nodea, nodeb):
    pass

def equals(nodea, nodeb):
    pass

def get_ordered_descendants(node, include_self=False):
    assert not include_self

def matching_descendants(nodea, nodeb, include_self=False):
    assert not include_self

    pairs = []

    i, j = 0, 0
    nodesa = get_ordered_descendants(nodea, include_self)
    nodesb = get_ordered_descendants(nodeb, include_self)
    na, nb = len(nodesa), len(nodesb)
    
    while i < na and j < nb:
        cmp = compare(nodesa[i], nodesb[j])
        if cmp > 0:
            j += 1
        elif cmp < 0:
            i += 1
        else: # cmp == 0, so the nodes match
            j_old = j
            while True:
                while True:
                    pair = nodesa[i], nodesb[j]
                    pairs.append(pair)
                    j += 1

                    if not (j < nb and equals(nodesa[i], nodesb[j])):
                        break

                i += 1
                j_final = j
                j = j_old

                if not (i < na and equals(nodesa[i], nodesb[j])):
                    break

            j = j_final
    
    return pairs
