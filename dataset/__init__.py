from . import utils
from .base import BaseDataset
from .plant_village import Plant_Village
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from common.Struct import Struct
import torchvision.transforms as transforms
from PIL import Image


_type = {
    'plant_village': Plant_Village
}


def load(name, root, classes, transform = None, img_size = 224, ind=None):
    ds = _type[name](root = root, classes = classes, transform = transform)
    ds.name = name
    return ds


def load_source_ds(params, config):
    data_path = config['dataset'][params.ds]['root']
    if not hasattr(params, 'source_classes'):
        params.source_classes = config['dataset'][params.ds]['classes']['source']

    img_size = config['transform_parameters']['sz_crop']
    ds_train = load(params.ds, data_path, classes=params.source_classes, img_size=img_size,
                    transform=utils.make_transform(**config['transform_parameters']))
    ds_val = load(params.ds, data_path, classes=params.source_classes, img_size=img_size,
                  transform=utils.make_transform(**config['transform_parameters'], is_train=False))

    n = len(ds_train)
    split = int((1-params.test_split) * n)  # train_size
    n_val = int((params.val_split * split))  # validation size
    n_train = split - n_val
    n_test = n - split
    indices = list(range(n))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = (
        indices[:n_train],
        indices[n_train:split],
        indices[split:],
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=params.batch_size, drop_last=True,
                                               pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=params.batch_size, shuffle=False, drop_last=True,
                                             pin_memory=True, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(ds_val, batch_size=params.batch_size, shuffle=False, pin_memory=True,
                                              sampler=test_sampler)

    return train_loader, val_loader, test_loader


def load_ds(params, config):
    data_path = config['dataset'][params.ds]['root']
    if not hasattr(params, 'source_classes'):
        params.source_classes = config['dataset'][params.ds]['classes']['source']
    if not hasattr(params, 'target_classes'):
        params.target_classes = config['dataset'][params.ds]['classes']['target']

    img_size = config['transform_parameters']['sz_crop']
    ds_train = load(params.ds, data_path, classes=params.source_classes, img_size=img_size,
                            transform=utils.make_transform(**config['transform_parameters']) )

    ds_test = load(params.ds, data_path, classes=params.target_classes,img_size=img_size,
                           transform=utils.make_transform(**config['transform_parameters'], is_train=False))
    n_test = len(ds_test)
    n_q = int(params.query_split * n_test)
    query_set, gallery_set = torch.utils.data.random_split(ds_test, [n_q, n_test-n_q])

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                               pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=params.batch_size, shuffle=False, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=params.batch_size, shuffle=False,
                                                 pin_memory=True)

    test_loader = Struct()
    test_loader.query, test_loader.gallery = query_loader, gallery_loader

    return train_loader, test_loader

def load_ds2(params, config):
    data_path = config['dataset'][params.ds]['root']
    if not hasattr(params, 'source_classes'):
        params.source_classes = config['dataset'][params.ds]['classes']['source']
    if not hasattr(params, 'target_classes'):
        params.target_classes = config['dataset'][params.ds]['classes']['target']

    img_size = config['transform_parameters']['sz_crop']
    ds_train = load(params.ds, data_path, classes=params.source_classes, img_size=img_size,
                            transform=utils.make_transform(**config['transform_parameters']) )

    ds_test = load(params.ds, data_path, classes=params.target_classes,img_size=img_size,
                           transform=utils.make_transform(**config['transform_parameters'], is_train=False))

    n_test = len(ds_test)
    if params.g_ipc == 1:
        gallery_idx = [391, 1955, 2255, 5102, 1174, 4442]
        query_idx = list(set(range(0, n_test)) - set(gallery_idx))
    else:
        n_gallery = params.g_ipc * len(params.target_classes)
        n_query =  n_test - n_gallery
        from sklearn.model_selection import train_test_split
        gallery_idx, query_idx = train_test_split(
            np.arange(len(ds_test)), test_size=n_query/n_test, shuffle=True, stratify=ds_test.ys)

    query_sampler = torch.utils.data.SubsetRandomSampler(query_idx)
    gallery_sampler = torch.utils.data.SubsetRandomSampler(gallery_idx)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                               pin_memory=True)
    query_loader = torch.utils.data.DataLoader(ds_test, batch_size=params.batch_size, shuffle=False,
                                               pin_memory=True, sampler=query_sampler)
    gallery_loader = torch.utils.data.DataLoader(ds_test, batch_size=params.batch_size, shuffle=False,
                                                 pin_memory=True, sampler=gallery_sampler)
    test_loader = Struct()
    test_loader.query, test_loader.gallery = query_loader, gallery_loader

    return train_loader, test_loader



def RGB2BGR(im):
    assert im.mode == 'RGB' or im.mode == 'RGBA'
    r, g, b = [im.getchannel(i) for i in range(3)]
    return Image.merge('RGB', (b, g, r))


def load_ds3(params, config):
    data_path = config['dataset'][params.ds]['root']
    if not hasattr(params, 'source_classes'):
        params.source_classes = config['dataset'][params.ds]['classes']['source']
    if not hasattr(params, 'target_classes'):
        params.target_classes = config['dataset'][params.ds]['classes']['target']

    normalize = transforms.Normalize(mean=[104., 117., 128.], std=[1., 1., 1.])

    train_transform = transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
            normalize,
        ])

    test_transform= transforms.Compose([
        transforms.Lambda(RGB2BGR),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        normalize,
   ])
   #
    ds_train = load(params.ds, data_path, classes=params.source_classes, img_size=256,
                            transform=train_transform)

    ds_test = load(params.ds, data_path, classes=params.target_classes,img_size=256, transform=test_transform)
    n_test = len(ds_test)
    if hasattr(params, 'gallery_split'):
        params.query_split = 1 - params.gallery_split
    n_q = int(params.query_split * n_test)
    query_set, gallery_set = torch.utils.data.random_split(ds_test, [n_q, n_test-n_q])

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=params.batch_size, shuffle=True, drop_last=True,
                                               pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=params.batch_size, shuffle=False, pin_memory=True)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=params.batch_size, shuffle=False,
                                                 pin_memory=True)

    test_loader = Struct()
    test_loader.query, test_loader.gallery = query_loader, gallery_loader

    return train_loader, test_loader