/**
Util functions for simple parsing
*/

export function generate_ids_from_value(valueArray, global_identifier=''){
	let myArr = [];
	if (valueArray != null ){
		if (checkIfArrayIsUnique(valueArray)){

			for (let i = 0; i < valueArray.length; i++) {
				let tempID = generate_ids_from_string(global_identifier + "_" + valueArray[i])
				myArr.push({"key": tempID, "value": valueArray[i]});
			}

			return myArr;
		}else throw new notUniqueException("IDs in config are not unique!");
	}
}

export function generate_ids_from_string(stringValue) {
	if (stringValue)	{
			let modString = stringValue.toString();
			modString = modString.toLowerCase();
			modString = modString.trim();
			modString = modString.replace(/[&\/\\#,+()$!§/&~%.'":*?<>{}]/g,'');
			modString = modString.replace(/\s+/g, "_");

			return modString;
	}else {
		console.log("No proper string for ID generation.");
		return null;
	}
}

export function generate_identifier(old_identifier, new_identifier) {

	var identifier = old_identifier + " " + new_identifier
	identifier = generate_ids_from_string(identifier)

	return identifier
}

function checkIfArrayIsUnique(myArray) {
	return myArray.length === new Set(myArray).size;
}

function notUniqueException(message) {
	this.message = message;
	this.name = "notUniqueException";
}