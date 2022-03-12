var loadFile = function(event) {
    if (document.getElementById("file").value != "") {

        var reader = new FileReader();
        reader.onload = function() {
            var output = document.getElementById('output');

            output.src = reader.result;

        };
        reader.readAsDataURL(event.target.files[0]);
    }

};

function myFunction() {
    var selected_file = document.getElementById("myfile");
    //console.log(typeof(selected_file));
    //console.log(selected_file);
    var selected_file_path = selected_file.value;
    console.log(selected_file_path);
    document.getElementById("2828").src = "file:///C:/Users/canara/Desktop/minerals_dataset/test_set/diamond/diamond.41.jpg";
    console.log("image uploaded..");
}