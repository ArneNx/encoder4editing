

def find_prefix(keys: list, p_agree: float = 0.66, separator="."):
    """
    Finds common prefix among state_dict keys
    :param keys: list of strings to find a common prefix
    :param p_agree: percentage of keys that should agree for a valid prefix
    :param separator: string that separates keys into substrings, e.g. "model.conv1.bias"
    :return: (prefix, end index of prefix)
    """
    keys = [k.split(separator) for k in keys]
    p_len = 0
    common_prefix = ""
    prefs = {"": len(keys)}
    while True:
        sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True)
        # check if largest count is above threshold
        if not prefs or sorted_prefs[0][1] < p_agree * len(keys):
            break
        common_prefix = sorted_prefs[0][0]  # save prefix

        p_len += 1
        prefs = {}
        for key in keys:
            if p_len == len(key):  # prefix cannot be an entire key
                continue
            p_str = ".".join(key[:p_len])
            prefs[p_str] = prefs.get(p_str, 0) + 1
    return common_prefix, p_len - 1


def load_state_dict(
    model,
    state_dict: dict,
    ignore_missing: bool = False,
    ignore_unused: bool = False,
    match_names: bool = False,
    ignore_dim_mismatch: bool = False,
    prefix_agreement: float = 0.98,
):
    """
    Loads given state_dict into model, but allows for some more flexible loading.

    :param model: nn.Module object
    :param state_dict: dictionary containing a whole state of the module (result of `some_model.state_dict()`)
    :param ignore_missing: if True ignores entries present in model but not in `state_dict`
    :param match_names: if True tries to match names in `state_dict` and `model.state_dict()`
                        by finding and removing a common prefix from the keys in each dict
    :param ignore_dim_mismatch: if True ignores parameters in `state_dict` that don't fit the shape in `model`
    """
    model_dict = model.state_dict()
    # 0. Try to match names by adding or removing prefix:
    if match_names:
        # find prefix keys of each state dict:
        s_pref, s_idx = find_prefix(list(state_dict.keys()), p_agree=prefix_agreement)
        m_pref, m_idx = find_prefix(list(model_dict.keys()), p_agree=prefix_agreement)
        # switch prefixes:
        stripped_state_dict = {}
        for k, v in state_dict.items():
            if k.split(".")[:s_idx] == s_pref.split("."):
                stripped_key = ".".join(k.split(".")[s_idx:])
            else:
                stripped_key = k
            new_key = m_pref + "." + stripped_key if m_pref else stripped_key
            stripped_state_dict[new_key] = v
        state_dict = stripped_state_dict

    # 1. filter out missing keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    unused = set(state_dict.keys()) - set(filtered_state_dict.keys())
    if unused and ignore_unused:
        print("Ignored unnecessary keys in pretrained dict:\n" + "\n".join(unused))
    elif unused:
        raise RuntimeError(
            "Error in loading state_dict: Unused keys:\n" + "\n".join(unused)
        )
    missing = set(model_dict.keys()) - set(filtered_state_dict.keys())
    if missing and ignore_missing:
        print("Ignored Missing keys:\n" + "\n".join(missing))
    elif missing:
        raise RuntimeError(
            "Error in loading state_dict: Missing keys:\n" + "\n".join(missing)
        )

    # 2. overwrite entries in the existing state dict
    updated_model_dict = {}
    for k, v in filtered_state_dict.items():
        if v.shape != model_dict[k].shape:
            if ignore_dim_mismatch:
                print("Ignored shape-mismatched parameter:", k)
                continue
            else:
                raise RuntimeError(
                    "Error in loading state_dict: Shape-mismatch for key {}".format(k)
                )
        updated_model_dict[k] = v

    # 3. load the new state dict
    model.load_state_dict(updated_model_dict, strict=(not ignore_missing))
