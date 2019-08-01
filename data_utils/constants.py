

class Constants(object):
    HDFS_BASE_PATH = "chenyuxiang/fm_rank/"
    BUSINESS_TYPE = "wap"
    TRAIN_DATA_PATH = HDFS_BASE_PATH + "train_data/"
    MODEL_PATH = HDFS_BASE_PATH + "model/"
    LOG_PATH = HDFS_BASE_PATH + "log/"
    APPNAME_PREFIX = "mp-rec-rank-service:"
    TABLE_DATA_PREFIX = "related_sample_"
    separatorForFMRedis = "#"
    clippingThreshold = 20.0
    expThreshold = 20.0
    negativeSampleRate = 1.0
    exposureExtendPosition = 40
    useDeltaNDCG = False
    pairwiseGradPercent = 1.0
    fieldDelimiter = ","
    crossFeaturesValueJoiner = "_"
    crossFeaturesKeyJoiner = "__"
    defaultPosValue = 101
    INVALID_HASH_VAL = -1
    MINVALUE_FOR_SAVE = 0.0 #hardcode
    countingFilterMaxValue = 100
    defaultDNNLayerSize = [128, 96, 1]
    denominatorSmooth= 1.0E-16 #Double.MinPositiveValue

    Pairwise = "pairwise"
    Pointwise = "pointwise"
    TrainType = set(Pairwise, Pointwise)
    catCols=[
                                 #"hour",
    #"pos",
    #"media_id",
      "news_channel_id",
      "news_category_id",
      "news_tags",
      "news_from",
      "news_type",
      "news_tag_list",
      "media_type",
      "media_level",
      "referred_news_channel_id",
      "referred_news_category_id",
      "referred_news_tags",
      "referred_news_tag_list",

    #cross feature
    "news_channel_id__referred_news_channel_id",
    "news_tag_list__referred_news_tag_list",
    "news_tags__referred_news_tags",
    "news_category_id__referred_news_category_id"
    ]


    FMV0catCols=[
    "referred_news_channel_id",
    "referred_news_category_id",
    "referred_news_from",
    "referred_news_type",
    "referred_news_priority",
    "referred_news_tag_list",

    #rec articles feature 6
    "news_channel_id",
    "news_category_id",
    "news_from",
    "news_type",
    "news_priority",
    "news_tag_list",

    #rec articles media 6
    "media_type",
    "media_from",
    "media_channel_id",
    "media_level",
    "media_audit_type",
    "media_id",

    #cross feature 3
    "news_channel_id__referred_news_channel_id",
    "news_category_id__referred_news_category_id"
    ]
    FMV1catCols=[
                                     #context  6
    "hour",
    "vst_province_id",
    "vst_city_id",
    "vst_device_id",
    "vst_os_id",
    "vst_browser_id",

    #referred article 6
    "referred_news_channel_id",
    "referred_news_category_id",
    "referred_news_from",
    "referred_news_type",
    "referred_news_priority",
    "referred_news_tag_list",

    #rec articles feature 6
    "news_channel_id",
    "news_category_id",
    "news_from",
    "news_type",
    "news_priority",
    "news_tag_list",

    #rec articles media 6
    "media_type",
    "media_from",
    "media_channel_id",
    "media_level",
    "media_audit_type",
    "media_id",

    #cross feature 3
    "news_channel_id__referred_news_channel_id",
    "news_category_id__referred_news_category_id"
    ]

    FMV2catCols=[
        "refer_mp_id", "mp_id"
    ]#.distinct

    FMV3catCols=[
        "news_post_time_gap_day",
        "referred_news_post_time_gap_day",
        "news_post_time_gap_hour",
        "referred_news_post_time_gap_hour"
    ]#.distinct

    FMV4catCols=[
        "news_title_len", "news_image_num", "news_cover_color_score", "news_cover_focus_score",
        "referred_news_title_len", "referred_news_image_num", "referred_news_cover_color_score",
        "referred_news_cover_focus_score"
    ]

    FMV5catCols=[
        "news_titleSegmentation", "referred_news_titleSegmentation",
        "news_titleSegmentation__referred_news_titleSegmentation"
    ]

    pairCatCols=[
                                     #"vst_province_id",
    #"vst_city_id",
    #"vst_device_id",
    #"vst_os_id",
    #"vst_browser_id",
    #"refer_mp_id",

    #"referred_news_post_time",
      "referred_news_channels",
      "referred_news_channel_id",
      "referred_news_category_id",
      "referred_news_tags",
    #"referred_news_original_source",
    #"referred_news_from",
    #"referred_news_cover",
    #"referred_news_title",
    #"referred_news_type",
    #"referred_news_audit_status",
    #"referred_news_priority",
    #"referred_news_images",
    #"referred_news_origin",
      "referred_news_tag_list",

    #"mp_id_1",
    #"news1_post_time",
      "news1_channels",
      "news1_channel_id",
      "news1_category_id",
      "news1_tags",
    #"news1_original_source",
    #"news1_from",
    #"news1_cover",
    #"news1_title",
      "news1_type",
    #"news1_audit_status",
    # "news1_priority",
    # "news1_images",
    #"news1_origin",
      "news1_tag_list",

    #"media1_id",
      "media1_type",
      "media1_from",
      "media1_channel_id",
      "media1_level",
    #"media1_publish_ditch",
    #"media1_is_original",
    #"media1_audit_type",

      "news1_channel_id__referred_news_channel_id",
      "news1_category_id__referred_news_category_id",
      "news1_tag_list__referred_news_tag_list",
      "news1_tags__referred_news_tags",

    #"mp_id_2",
    #"news2_post_time",
      "news2_channels",
      "news2_channel_id",
      "news2_category_id",
      "news2_tags",
    #"news2_original_source",
    #"news2_from",
    #"news2_cover",
    #"news2_title",
      "news2_type",
    #"news2_audit_status",
    #"news2_priority",
    #"news2_images",
    #"news2_origin",
      "news2_tag_list",

    #"media2_id",
      "media2_type",
      "media2_from",
      "media2_channel_id",
      "media2_level",
    #"media2_publish_ditch",
    #"media2_is_original",
    #"media2_audit_type",

      "news2_channel_id__referred_news_channel_id",
      "news2_category_id__referred_news_category_id",
      "news2_tag_list__referred_news_tag_list",
      "news2_tags__referred_news_tags"
    ]

    numCols=[
    ]

    FMnumCols=[
    ]


