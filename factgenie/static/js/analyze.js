const metadata = window.metadata;
const campaigns = window.campaigns;

function deleteRow(button) {
    $(button).parent().parent().remove();
}

function getSelectedCampaigns() {
    const selectedCampaigns = $('.btn-check-campaign:checked').map(function () {
        return $(this).data('content');
    }).get();
    return selectedCampaigns;
}

function downloadBlob(data, filename) {
    const blob = new Blob([data], { type: 'application/zip' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'agreement_files.zip';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function downloadIaaFiles() {
    // get the selected campaigns and ask the backend
    const selectedCombinations = gatherSelectedCombinations();
    const selectedCampaigns = getSelectedCampaigns();

    if (selectedCombinations.length == 0) {
        alert("Please select some data for comparison.");
        return;
    }

    $("#agreement-spinner").show();

    $.post({
        url: `${url_prefix}/compute_agreement`,
        contentType: 'application/json', // Specify JSON content type
        data: JSON.stringify({
            selectedCampaigns: selectedCampaigns,
            combinations: selectedCombinations,
        }),
        xhrFields: {
            responseType: 'blob'  // Set response type to blob for binary data
        },
        success: function (response) {
            downloadBlob(response, 'agreement_files.zip');
            $("#agreement-spinner").hide();
        },
        error: function (response) {
            alert("An error occurred: " + response.responseText);
        }
    });
}


function populateTable(tableId, data, columns) {
    var table = $(`#${tableId} tbody`);
    table.empty(); // Clear existing rows

    for (const obj of data) {
        let row = '<tr>';
        for (const col of columns) {
            if (col === 'annotation_type') {
                row += `<td>
                    <span class="badge" style="background-color: ${metadata.config.annotation_span_categories[obj[col]].color}; color: rgb(253, 253, 253);">
                        ${metadata.config.annotation_span_categories[obj[col]].name}
                    </span>
                </td>`;
            } else {
                row += `<td>${obj[col]}</td>`;
            }
        }
        row += '</tr>';
        table.append(row);
    }
    $(`#${tableId}`).bootstrapTable();
}

function updateComparisonData() {
    const selectedCampaigns = getSelectedCampaigns();

    $('#common-categories').html("None");
    $('#common-examples').html("0");
    $("#selectedDatasetsContent").empty();
    $("#agreement-btn").addClass("disabled")

    // find which category label names are common to all campaigns
    const campaignCategories = selectedCampaigns.map(c => campaigns[c].metadata.config.annotation_span_categories).map(c => c.map(cat => cat.name));
    const commonCategories = campaignCategories.reduce((acc, val) => {
        return acc.filter(x => val.some(y => y === x));
    });

    $('#common-categories').html(
        commonCategories.map(c => `<span class="badge bg-secondary">${c}</span>`).join("\n")
    );

    // Create campaign-annotator group combinations
    const campaignAnnotatorGroups = selectedCampaigns.flatMap(campaign => {
        const campaignData = campaigns[campaign].data;
        const annotatorGroups = [...new Set(campaignData.map(d => d.annotator_group))];
        return annotatorGroups.map(group => ({ campaign, group }));
    });

    // Get examples for each campaign-annotator group combination
    const combinations = campaignAnnotatorGroups.map(({ campaign, group }) =>
        campaigns[campaign].data.filter(d => d.annotator_group === group)
    );

    // Find common examples across all combinations
    const commonExamples = combinations.reduce((acc, val) => {
        return acc.filter(x => val.some(y =>
            y.dataset === x.dataset &&
            y.split === x.split &&
            y.setup_id === x.setup_id
        ));
    });
    const finishedExamples = commonExamples.filter(e => e.status === 'finished');

    // Count examples per dataset-split-setup combination
    const exampleCounts = finishedExamples.reduce((acc, val) => {
        const key = `${val.dataset}|${val.split}|${val.setup_id}`;
        acc[key] = (acc[key] || 0) + 1;
        return acc;
    }, {});

    const filteredExampleCounts = Object.entries(exampleCounts).reduce((acc, [key, count]) => {
        const [dataset, split, setup_id] = key.split('|');
        const groupsWithExample = campaignAnnotatorGroups.filter(({ campaign, group }) => {
            return campaigns[campaign].data.some(d =>
                d.annotator_group === group &&
                d.dataset === dataset &&
                d.split === split &&
                d.setup_id === setup_id &&
                d.status === 'finished'
            );
        });

        if (groupsWithExample.length >= 2) {
            acc[key] = count;
        }
        return acc;
    }, {});

    const comparisonData = Object.entries(filteredExampleCounts).map(([key, count]) => {
        const [dataset, split, setup_id] = key.split('|');
        const groups = campaignAnnotatorGroups
            .map(({ campaign, group }) => `${campaign}:${group}`)
            .join(", ");
        return { dataset, split, setup_id, example_count: count, groups };
    });

    $("#selectedDatasetsContent").html(
        comparisonData.map(d =>
            `<tr>
                <td>${d.dataset}</td>
                <td>${d.split}</td>
                <td>${d.setup_id}</td>
                <td>${d.example_count}</td>
                <td><small>${d.groups}</small></td>
                <td><button type="button" class="btn btn-sm btn-secondary" onclick="deleteRow(this);">x</button></td>
            </tr>`
        ).join("\n")
    );

    $('#common-examples').html(finishedExamples.length);
    $("#agreement-btn").removeClass("disabled");

    return combinations;
}

const fullTableColumns = ['dataset', 'split', 'setup_id', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
const spanTableColumns = ['example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
const setupTableColumns = ['setup_id', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];
const datasetTableColumns = ['dataset', 'split', 'example_count', 'annotation_type', 'ann_count', 'avg_count', 'prevalence'];


$(document).ready(function () {
    // if we are on a detail page, populate the tables
    if ($('#full-table').length > 0) {
        const statistics = window.statistics;
        const ann_counts = statistics.ann_counts;

        populateTable('full-table', ann_counts.full, fullTableColumns);
        populateTable('span-table', ann_counts.span, spanTableColumns);
        populateTable('setup-table', ann_counts.setup, setupTableColumns);
        populateTable('dataset-table', ann_counts.dataset, datasetTableColumns);
    }
});


$(document).on('change', '.btn-check-campaign', updateComparisonData);