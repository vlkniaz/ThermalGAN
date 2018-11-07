def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'thermal_gan':
        assert(opt.dataset_mode == 'thermal')
        from .thermal_gan_model import ThermalGANModel
        model = ThermalGANModel()
    elif opt.model == 'thermal_gan_rel':
        assert(opt.dataset_mode == 'thermal_rel')
        from .thermal_gan_rel_model import ThermalGANRelModel
        model = ThermalGANRelModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
