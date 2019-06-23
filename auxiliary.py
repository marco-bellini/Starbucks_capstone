
import pandas as pd


def split_offers_before(offers, time_split):
    """
    split_offers_before

    :param offers:
    :param time_split:
    :return:
    """

    s_offers = offers.loc[(offers.offer_received < time_split)]
    return (s_offers)


def split_offers_after(offers, time_split):
    """
    split_offers_after
    :param offers:
    :param time_split:
    :return:
    """
    s_offers = offers.loc[(offers.offer_end > time_split)]
    return (s_offers)


def find_p_o(df, p, o):
    """
    find_p_o

    :param df:
    :param p:
    :param o:
    :return:
    """
    return (df.loc[((df.person == p) & (df.offer == o))])