from collections import OrderedDict
import os


def load_params(args, n_s, df_train, scaler, mu_gmm=None, sigma_gmm=None):

    params = OrderedDict(
        seed=args.seed,
        run_name=args.run_name,
        epsilon=args.epsilon,
        delta=args.delta,
        n_s=n_s,
    )

    if ('GAUSSIAN' in args.model_class) or ('UNIFORM' in args.model_class):
        params.update(
            OrderedDict(
                df_=df_train,
                gmm=(args.fn_csv=='gmm'),
                mu_gmm=mu_gmm,
                sigma_gmm=sigma_gmm,
                scaler=scaler,
        )
    )

    # Model family specific params
    if "GAN" in args.model_class or "VAE" in args.model_class:
        params.update(
            OrderedDict(
                df=df_train,
                batch_size=args.batch_size,
                save=args.save,
                load=args.load,
            )
        )
    if args.model_class in ["PATEGAN", "PATECTGAN"]:
        params.update(
            OrderedDict(
                sample_per_teacher=args.sample_per_teacher,
                laplace_scale=args.laplace_scale,
                moments_order=args.moments_order,
            )
        )
    if args.model_class in ["DPCTGAN", "PATECTGAN", "DPTVAE"]:
        params.update(
            OrderedDict(
                categorical_columns=args.categorical_columns.split(",") if args.categorical_columns else tuple(),
                ordinal_columns=args.ordinal_columns.split(",") if args.ordinal_columns else tuple(),
            )
        )
    if args.model_class in ["DPGAN", "DPCTGAN", "DPTVAE"]:
        params.update(
            OrderedDict(
                iterations=args.iterations,
                noise_multiplier=args.noise_multiplier,
            )
        )
    if args.model_class == "PrivBayes":
        params.update(
            OrderedDict(
                x_train=df_train,
                category_threshold=args.category_threshold,
                k=args.num_parents,
                histogram_bins=args.histogram_bins,
                other_categorical=args.other_categorical,
                keys=args.keys,
                in_path=args.fn_csv,
                out_dir=args.out_dir,
            )
        )
    if args.model_class == "RONGauss":
        params.update(
            OrderedDict(
                df=df_train,
                algorithm=args.algorithm,
                dimension=args.dimension,
                target=args.target,
            )
        )

    # UNUSED ?
    # if args.model_class == "pategan":
    #     params.update(
    #         OrderedDict(
    #             x_train=np_train,
    #             batch_size=args.batch_size,
    #             lamda=args.lamda,
    #             num_students=args.num_students,
    #             num_teachers=args.num_teachers,
    #         )
    #     )

    return params


def log(params, task, loss, weight, resampl, alpha=1):
    if not os.path.exists(f'logs/logs.txt'):
        with open(f'logs/logs.txt', 'a+') as file:
            file.write('; '.join([k for k, v in params.items() if k not in ['df']]) + \
            'weight' + ';' + 'resampl' + ';' + 'task' + \
            ';' + 'loss' + ';' + 'alpha' + '\n')

    with open(f'logs/logs.txt', 'a+') as file:
        file.write('; '.join([str(v) for k, v in params.items() if k not in ['df']]) + \
            weight + ';' + resampl + ';' + task + \
            ';' + str(round(loss, 4)) + ';' + str(alpha) + '\n')
