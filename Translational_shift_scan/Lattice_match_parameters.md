zsl = ZSLG(max_area=350, #350
           max_length_tol=0.02, #0.02
           max_angle_tol=0.009,
           max_area_ratio_tol=0.20)

builder = CIB(film_structure=hfo2_str,
              substrate_structure=sto_str,
              film_miller=(-1, 1, 1),
              substrate_miller=(1, 0, 0),
              zslgen=zsl,
              filter_out_sym_slabs=False,
              label_index=True)

print("Available terminations:", builder.terminations)

termination = ('2_O2_P-1_1', '1_SrO_P4/mmm_2')
if termination not in builder.terminations:
    raise ValueError(
        f"target termination {termination} not found; available "
        f"terminations for this run: {builder.terminations}")

interfaces = list(builder.get_interfaces(
    termination=termination,
    gap=2.5,
    film_thickness=5,
    substrate_thickness=3,
    in_layers=True,
))
