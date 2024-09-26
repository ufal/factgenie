const url_prefix = window.url_prefix;


function deleteCampaign(campaignId, source) {
    // ask for confirmation
    if (!confirm(`Are you sure you want to delete the campaign ${campaignId}? All the data will be lost!`)) {
        return;
    }

    $.post({
        url: `${url_prefix}/delete_campaign`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            campaignId: campaignId,
            source: source,
            mode: window.mode
        }),
        success: function (response) {
            console.log(response);

            if (response.success !== true) {
                alert(response.error);
            } else {
                // remove the campaign from the list
                $(`#campaign-${campaignId}`).remove();

                // reload the page
                location.reload();
            }
        }
    });
}