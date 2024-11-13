const datasets = window.datasets;
const url_prefix = window.url_prefix;
const mode = window.mode;

function addAnnotationSpanCategory() {
    const annotationSpanCategories = $("#annotation-span-categories");
    const randomColor = '#' + Math.floor(Math.random() * 16777215).toString(16);
    const newCategory = createAnnotationSpanCategoryElem("", randomColor, "");
    annotationSpanCategories.append(newCategory);
}

function addModelArgument() {
    const modelArguments = $("#model-arguments");
    const newArg = createArgElem("", "");
    modelArguments.append(newArg);
}

function addExtraArgument() {
    const modelArguments = $("#extra-arguments");
    const newArg = createArgElem("", "");
    modelArguments.append(newArg);
}

function addFlag() {
    const flags = $("#flags");
    const newFlag = createFlagElem("");
    flags.append(newFlag);
}

function addOption() {
    const options = $("#options");
    const newOption = createOptionElem("select", "", "");
    options.append(newOption);
}

function addTextField() {
    const textFields = $("#textFields");
    const newTextField = createTextFieldElem("");
    textFields.append(newTextField);
}


function deleteRow(button) {
    $(button).parent().parent().remove();
}

function createFlagElem(key) {
    // text area and selectbox for the flag ("checked" or "unchecked" based on the value)
    const newFlag = $(`
        <div class="row mt-1">
        <div class="col-11">
        <input type="text" class="form-control" name="argName" value="${key}" placeholder="Question or statement">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newFlag;
}

function createOptionElem(type, label, values) {
    // three columns: option type (selectbox, slider) text input for the label, and text input for comma-separated values
    const newOption = $(`
        <div class="row mt-1">
        <div class="col-3">
        <select class="form-select" name="optionType">
            <option value="select" ${type === 'select' ? 'selected' : ''}>Select box</option>
            <option value="slider" ${type === 'slider' ? 'selected' : ''}>Slider</option>
        </select>
        </div>
        <div class="col-3">
        <input type="text" class="form-control" name="optionLabel" value="${label}" placeholder="Label">
        </div>
        <div class="col-5">
        <input type="text" class="form-control" name="optionValues" value="${values}" placeholder="Comma-separated values">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        `);
    return newOption;
}

function createTextFieldElem(key) {
    // text area and selectbox for the flag ("checked" or "unchecked" based on the value)
    const newFlag = $(`
        <div class="row mt-1">
        <div class="col-11">
        <input type="text" class="form-control" name="argName" value="${key}" placeholder="Text field label">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newFlag;
}

function createArgElem(key, value) {
    // escape quotes in the value
    if (typeof value === 'string') {
        value = value.replace(/"/g, "&quot;");
    }
    const newArg = $(`
        <div class="row mt-1">
        <div class="col-6">
        <input type="text" class="form-control"  name="argName" value="${key}" placeholder="Key">
        </div>
        <div class="col-5">
        <input type="text" class="form-control" name="argValue" value="${value}" placeholder="Value">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newArg;
}

function createAnnotationSpanCategoryElem(name, color, description) {
    const newCategory = $(`
        <div class="row mt-1">
        <div class="col-3">
        <input type="text" class="form-control" name="annotationSpanCategoryName" value="${name}" placeholder="Category name">
        </div>
        <div class="col-1">
        <input type="color" class="form-control" name="annotationSpanCategoryColor" value="${color}">
        </div>
        <div class="col-7">
        <input type="text" class="form-control" name="annotationSpanCategoryDescription" placeholder="Description" value="${description}">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newCategory;
}


function detailFormatter(index, row) {
    const keys = Object.keys(row).filter(key => key.match(/^\d+$/));
    const key = keys[keys.length - 1];

    return row[key];
}

function detailFilter(index, row) {
    // for all key value pairs in row, check if we can make a jquery object out of the value
    // if we can, check if the text of the object is "finished"
    // if it is, return true

    for (const key in row) {
        const value = row[key];
        try {
            const $value = $(value);
            if ($value.text().trim() === "finished") {
                return true;
            }
        } catch (e) {
            // pass
        }
    }
}

function enableTooltips() {
    // enable tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })
}


function gatherSelectedCombinations() {
    // read all the rows the remained in #selectedDatasetsContent
    var selectedData = [];
    $("#selectedDatasetsContent tr").each(function () {
        var dataset = $(this).find("td:eq(0)").text();
        var split = $(this).find("td:eq(1)").text();
        if (mode != "llm_gen") {
            var setup_id = $(this).find("td:eq(2)").text();
            selectedData.push({ dataset: dataset, split: split, setup_id: setup_id });
        } else {
            selectedData.push({ dataset: dataset, split: split });
        }
    });
    return selectedData;
}


function getAnnotationSpanCategories() {
    var annotationSpanCategories = [];

    $("#annotation-span-categories").children().each(function () {
        const name = $(this).find("input[name='annotationSpanCategoryName']").val();
        const color = $(this).find("input[name='annotationSpanCategoryColor']").val();
        const description = $(this).find("input[name='annotationSpanCategoryDescription']").val();
        annotationSpanCategories.push({ name: name, color: color, description: description });
    });
    return annotationSpanCategories;
}

function getKeysAndValues(div) {
    var args = {};
    div.children().each(function () {
        const key = $(this).find("input[name='argName']").val();
        const value = $(this).find("input[name='argValue']").val();
        args[key] = value;
    });
    return args;
}

function getKeys(div) {
    var keys = [];
    div.children().each(function () {
        const key = $(this).find("input[name='argName']").val();
        keys.push(key);
    });
    return keys;
}

function getOptions() {
    var options = [];

    $("#options").children().each(function () {
        const type = $(this).find("select[name='optionType']").val();
        const label = $(this).find("input[name='optionLabel']").val();
        const values = $(this).find("input[name='optionValues']").val().split(",").map(v => v.trim());
        options.push({ type: type, label: label, values: values });
    });
    return options;
}

function mod(n, m) {
    return ((n % m) + m) % m;
}


function randInt(max) {
    return Math.floor(Math.random() * max);
}

function nextBtn() {
    goToPage(current_example_idx + 1);
}

function prevBtn() {
    goToPage(current_example_idx - 1);
}

function startBtn() {
    goToPage(0);
}

function endBtn() {
    goToPage(total_examples - 1);
}

function randomBtn() {
    goToPage(randInt(total_examples - 1));
}


function goToBtn() {
    var n = $("#page-input").val();
    goToPage(n);
}