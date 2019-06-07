def check_grid(grid):
    """Used to ensure that the given grid settings are valid.

    Args:
        grid(dict): Dictionary containing the grid setup.
    """
    msg = "The grid information is missing the value for {}"
    val_names = ["min", "max", "sigma", "n"]
    for val_name in val_names:
        try:
            grid[val_name]
        except Exception:
            raise KeyError(msg.format(val_name))

    # Make the n into integer
    grid["n"] = int(grid["n"])
    assert grid["min"] < grid["max"], \
        "The min value should be smaller than the max value."


class MBTRGrid(dict):
    """Custom class for storing and modifying the MBTR grid setup. Needed in
    order to keep track of changes to the grid spacing.
    """
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self.nupdated = False

    def __setitem__(self, key, item):
        super().__setitem__(key, item)
        if key == "n":
            self.nupdated = True


class MBTRSetup(dict):
    """Custom class for storing and modifying the MBTR setup. Needed in order
    to check the validity of changes.
    """
    def __init__(self, *arg, **kw):
        self.update(*arg, **kw)
        self.periodic = None

    def set_periodic(self, periodic):
        self.periodic = periodic

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError("update expected at most 1 arguments, "
                                "got %d" % len(args))
            other = dict(args[0])
            for key in other:
                self[key] = other[key]
        for key in kwargs:
            self[key] = kwargs[key]

    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        return self[key]


class MBTRK1Setup(MBTRSetup):
    """Custom class for storing and modifying the MBTR k=1 setup. Needed in
    order to check the validity of changes.
    """
    def __setitem__(self, key, value):
        # Check that only valid keys are used.
        valid_keys = set(("geometry", "grid"))
        if key not in valid_keys:
            raise ValueError("The given setup contains the following invalid key: {}".format(key))

        # Check geometry
        if key == "geometry":
            geom_func = value.get("function")
            if geom_func is not None:
                valid_geom_func = set(("atomic_number",))
                if geom_func not in valid_geom_func:
                    raise ValueError(
                        "Unknown geometry function specified for k=1. Please use one of"
                        " the following: {}".format(valid_geom_func)
                    )

        # Check the weighting function
        if key == "weighting":
            if value is not None:
                valid_weight_func = set(("unity",))
                weight_func = value.get("function")
                if weight_func not in valid_weight_func:
                    raise ValueError(
                        "Unknown weighting function specified for k=1. Please use one of"
                        " the following: {}".format(valid_weight_func)
                    )

        # Check grid
        if key == "grid":
            check_grid(value)
            value = MBTRGrid(value)

        # Set value after checks
        super().__setitem__(key, value)


class MBTRK2Setup(MBTRSetup):
    """Custom class for storing and modifying the MBTR k=2 setup. Needed in
    order to check the validity of changes.
    """
    def __setitem__(self, key, value):
        # Check that only valid keys are used.
        valid_keys = set(("geometry", "grid", "weighting"))
        if key not in valid_keys:
            raise ValueError("The given setup contains the following invalid key: {}".format(key))

        # Check geometry
        if key == "geometry":
            geom_func = value.get("function")
            if geom_func is not None:
                valid_geom_func = set(("distance", "inverse_distance"))
                if geom_func not in valid_geom_func:
                    raise ValueError(
                        "Unknown geometry function specified for k=2. Please use one of"
                        " the following: {}".format(valid_geom_func)
                    )

        # Check the weighting function
        if key == "weighting":
            if value is not None:
                valid_weight_func = set(("unity", "exponential", "exp"))
                weight_func = value.get("function")
                if weight_func not in valid_weight_func:
                    raise ValueError(
                        "Unknown weighting function specified for k=2. Please use one of"
                        " the following: {}".format(valid_weight_func)
                    )
                else:
                    if weight_func == "exponential" or weight_func == "exp":
                        needed = ("cutoff", "scale")
                        for pname in needed:
                            param = value.get(pname)
                            if param is None:
                                raise ValueError(
                                    "Missing value for '{}' in the k=2 weighting.".format(key)
                                )

        # Check grid
        if key == "grid":
            check_grid(value)
            value = MBTRGrid(value)

        # Set value after checks
        super().__setitem__(key, value)


class MBTRK3Setup(MBTRSetup):
    """Custom class for storing and modifying the MBTR k=3 setup. Needed in
    order to check the validity of changes.
    """
    def __setitem__(self, key, value):
        # Check that only valid keys are used.
        valid_keys = set(("geometry", "grid", "weighting"))
        if key not in valid_keys:
            raise ValueError("The given setup contains the following invalid key: {}".format(key))

        # Check geometry
        if key == "geometry":
            geom_func = value.get("function")
            if geom_func is not None:
                valid_geom_func = set(("angle", "cosine"))
                if geom_func not in valid_geom_func:
                    raise ValueError(
                        "Unknown geometry function specified for k=2. Please use one of"
                        " the following: {}".format(valid_geom_func)
                    )

        # Check the weighting function
        if key == "weighting":
            if value is not None:
                valid_weight_func = set(("unity", "exponential", "exp"))
                weight_func = value.get("function")
                if weight_func not in valid_weight_func:
                    raise ValueError(
                        "Unknown weighting function specified for k=2. Please use one of"
                        " the following: {}".format(valid_weight_func)
                    )
                else:
                    if weight_func == "exponential" or weight_func == "exp":
                        needed = ("cutoff", "scale")
                        for pname in needed:
                            param = value.get(pname)
                            if param is None:
                                raise ValueError(
                                    "Missing value for '{}' in the k=3 weighting.".format(key)
                                )

        # Check grid
        if key == "grid":
            check_grid(value)
            value = MBTRGrid(value)

        # Set value after checks
        super().__setitem__(key, value)
