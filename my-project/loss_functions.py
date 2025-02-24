import torch
from src.utils import list_of_distances, list_of_norms

def loss_f_default(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_ae, lambda_1, lambda_2):
    # softmax crossentropy loss - E
    loss_function = torch.nn.CrossEntropyLoss()
    train_ce = loss_function(pred_y, batch_y)

    prototype_distances = model.prototype_layer.prototype_distances
    feature_vectors = model.feature_vectors

    # intepretability loss R1
    train_e1 = torch.mean(torch.min(list_of_distances(prototype_distances, feature_vectors.view(-1, model.in_channels_prototype)), dim=1)[0])
    # intepretability loss R2
    train_e2 = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.in_channels_prototype ), prototype_distances), dim=1)[0])

    # autoencoder loss - A
    out_decoder = model.decoder(feature_vectors)
    train_ae = torch.mean(list_of_norms(out_decoder-elastic_batch_x))

    train_te = lambda_class * train_ce +\
            lambda_1 * train_e1 +\
            lambda_2 * train_e2 +\
            lambda_ae * train_ae
    
    return train_te, train_ce, train_ae, train_e1, train_e2