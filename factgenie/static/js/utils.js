const datasets = window.datasets;
const url_prefix = window.url_prefix;
const mode = window.mode;

function addAnnotationSpanCategory(name, description, customColor) {
    const annotationSpanCategories = $("#annotation-span-categories");

    name = name || "";
    description = description || "";

    const newCategory = createAnnotationSpanCategoryElem(name, description);
    annotationSpanCategories.append(newCategory);

    const colors = [
        '#9467bd', '#e377c2', '#e7298a', '#d62728',
        '#66a61e', '#1b9e77', '#1f77b4', '#024983',
        '#bcbd22', '#e6ab02', '#ff7f0e', '#d95f02',
        '#8c564b', '#a6761d', '#666666', '#7f7f7f'
    ];
    var elem = $('.color-input').last()[0];

    var hueb = new Huebee(elem, {
        notation: 'hex',
        customColors: colors,
        shades: 0,
        hues: 4,
        setText: false,
    });
    if (customColor) {
        hueb.setColor(customColor);
    } else {
        const randomColor = colors[Math.floor(Math.random() * colors.length)];
        hueb.setColor(randomColor);
    }
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
    const newOption = createOptionElem("", "");
    options.append(newOption);
}

function addSlider() {
    const sliders = $("#sliders");
    const newSlider = createSliderElem("", "", "", "");
    sliders.append(newSlider);
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

function createOptionElem(label, values) {
    // three columns: option type (selectbox, slider) text input for the label, and text input for comma-separated values
    const newOption = $(`
        <div class="row mt-1">
        <div class="col-4">
        <input type="text" class="form-control" name="optionLabel" value="${label}" placeholder="Label">
        </div>
        <div class="col-7">
        <input type="text" class="form-control" name="optionValues" value="${values}" placeholder="Comma-separated values">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        `);
    return newOption;
}

function createSliderElem(key, min, max, step) {
    // three columns: option type (selectbox, slider) text input for the label, and text input for comma-separated values
    const newSlider = $(`
        <div class="row mt-1">
        <div class="col-5">
        <input type="text" class="form-control" name="sliderLabel" value="${key}" placeholder="Label">
        </div>
        <div class="col-2">
        <input type="number" class="form-control" name="sliderMin" value="${min}" placeholder="Min">
        </div>
        <div class="col-2">
        <input type="number" class="form-control" name="sliderMax" value="${max}" placeholder="Max">
        </div>
        <div class="col-2">
        <input type="number" class="form-control" name="sliderStep" value="${step}" placeholder="Step">
        </div>
        <div class="col-1">
        <button type="button" class="btn btn-danger" onclick="deleteRow(this);">x</button>
        </div>
        </div>
    `);
    return newSlider;
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
function createAnnotationSpanCategoryElem(name, description) {
    const newCategory = $(`
        <div class="row mt-1">
        <div class="col-3">
        <input type="text" class="form-control" name="annotationSpanCategoryName" value="${name}" placeholder="Category name">
        </div>
        <div class="col-1">
        <a class="form-control color-input" name="annotationSpanCategoryColor"></a>
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
        const color = $(this).find("a[name='annotationSpanCategoryColor']").css('background-color');
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
        const label = $(this).find("input[name='optionLabel']").val();
        const values = $(this).find("input[name='optionValues']").val().split(",").map(v => v.trim());
        options.push({ label: label, values: values });
    });
    return options;
}

function getSliders() {
    var sliders = [];

    $("#sliders").children().each(function () {
        const label = $(this).find("input[name='sliderLabel']").val();
        const min = $(this).find("input[name='sliderMin']").val();
        const max = $(this).find("input[name='sliderMax']").val();
        const step = $(this).find("input[name='sliderStep']").val();

        sliders.push({ label: label, min: min, max: max, step: step });
    });
    return sliders;
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