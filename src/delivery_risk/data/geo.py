"""Координаты почтовых префиксов и расстояние между покупателем и продавцом.

Справочник geolocation содержит много точек на префикс; берётся медиана.
Если префикса в справочнике нет, используется центр штата - медиана всех
точек штата. Справочник не зависит от заказов, поэтому центры штатов
вычисляются по нему, а не по обучающей выборке.
"""
import numpy as np
import pandas as pd

GEO = "data/raw/olist/olist_geolocation_dataset.csv"
R_EARTH_KM = 6371.0


def load_reference(path=GEO):
    g = pd.read_csv(path, dtype={"geolocation_zip_code_prefix": str})
    # в справочнике есть точки вне Бразилии - ошибки ввода
    g = g[g["geolocation_lat"].between(-34, 6) & g["geolocation_lng"].between(-74, -34)]
    by_zip = g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()
    by_state = g.groupby("geolocation_state")[["geolocation_lat", "geolocation_lng"]].median()
    by_zip.columns = by_state.columns = ["lat", "lng"]
    return by_zip, by_state


def coords(zips, states, ref):
    by_zip, by_state = ref
    z = by_zip.reindex(zips.to_numpy())
    s = by_state.reindex(states.to_numpy())
    missing = z["lat"].isna().to_numpy()
    lat = np.where(missing, s["lat"].to_numpy(), z["lat"].to_numpy())
    lng = np.where(missing, s["lng"].to_numpy(), z["lng"].to_numpy())
    return lat, lng, missing


def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def distance_km(df, ref):
    """Расстояние покупатель - продавец и флаг «координаты по центру штата»."""
    la1, ln1, m1 = coords(df["customer_zip_code_prefix"], df["customer_state"], ref)
    la2, ln2, m2 = coords(df["seller_zip_code_prefix"], df["seller_state"], ref)
    return haversine(la1, ln1, la2, ln2), (m1 | m2)
