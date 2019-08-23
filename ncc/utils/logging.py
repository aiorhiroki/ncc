import requests


def slack_logging(
    file_name,
    token,
    channel,
    title,
    filename='Metric Figure'
):
    """
    slackに画像ファイルを送信します。
    """

    files = {'file': open(file_name, 'rb')}
    param = dict(
        token=token,
        channels=channel,
        filename=filename,
        title=title
    )
    requests.post(
        url='https://slack.com/api/files.upload',
        params=param,
        files=files
    )
