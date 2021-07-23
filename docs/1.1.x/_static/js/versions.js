// Open file
var xhr = new XMLHttpRequest();
xhr.onreadystatechange = () => {
  if(xhr.readyState === XMLHttpRequest.DONE) {
    let versions = JSON.parse(xhr.responseText).versions;
    let versionTable = document.getElementById("version-table");
    for (let i=0; i < versions.length; ++i) {
      let version = versions[i];
      let dd = document.createElement("dd");
      let link = document.createElement("a");
      link.href = `${version.url}`;
      link.textContent = `${version.versionNumber} ${version.description}`;
      dd.appendChild(link);
      versionTable.appendChild(dd);
    }
  }
};
xhr.open("GET", "https://singroup.github.io/dscribe/versions.json", true);
xhr.send();
