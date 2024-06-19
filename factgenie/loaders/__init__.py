from factgenie.loaders.ice_hockey import IceHockey
from factgenie.loaders.gsmarena import GSMArena
from factgenie.loaders.openweather import OpenWeather
from factgenie.loaders.owid import OurWorldInData
from factgenie.loaders.wikidata import Wikidata
from factgenie.loaders.logicnlg import LogicnlgDev100
from factgenie.loaders.dummy import Dummy

DATASET_CLASSES = {
    "ice_hockey": IceHockey,
    "gsmarena": GSMArena,
    "openweather": OpenWeather,
    "owid": OurWorldInData,
    "wikidata": Wikidata,
    "logicnlg": LogicnlgDev100,
    "dummy": Dummy,
}
