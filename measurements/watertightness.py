from misc import create_mesh
from cadmodel.model import CADModel


def is_watertight(pred: CADModel):
    pred_mesh = create_mesh(pred)

    return pred_mesh.is_watertight
