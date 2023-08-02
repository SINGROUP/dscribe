/*
	--------------------------------------------------------------------------
	$Id: spamspan.js 5 2007-09-29 15:56:26Z moltar $
	--------------------------------------------------------------------------
	Version: 1.03
	Release date: 13/05/2006
	Last update: 07/01/2007

	(c) 2006 SpamSpan (www.spamspan.com)

	This program is distributed under the terms of the GNU General Public
	Licence version 2, available at http://www.gnu.org/licenses/gpl.txt
	--------------------------------------------------------------------------
*/

var spamSpanMainClass		= 'spamspan';
var spamSpanUserClass		= 'u';
var spamSpanDomainClass		= 'd';
var spamSpanAnchorTextClass = 't';
var spamSpanParams			= new Array('subject', 'body');

/*
	--------------------------------------------------------------------------
	Do not edit past this point unless you know what you are doing.
	--------------------------------------------------------------------------
*/

// load SpamSpan
addEvent(window, 'load', spamSpan);

function spamSpan() {
	var allSpamSpans = getElementsByClass(spamSpanMainClass, document, 'span');
	for (var i = 0; i < allSpamSpans.length; i++) {
		// get data
		var user = getSpanValue(spamSpanUserClass, allSpamSpans[i]);
		var domain = getSpanValue(spamSpanDomainClass, allSpamSpans[i]);
		var anchorText = getSpanValue(spamSpanAnchorTextClass, allSpamSpans[i]);
		// prepare parameter data
		var paramValues = new Array();
		for (var j = 0; j < spamSpanParams.length; j++) {
			var paramSpanValue = getSpanValue(spamSpanParams[j], allSpamSpans[i]);
			if (paramSpanValue) {
				paramValues.push(spamSpanParams[j] + '=' +
					encodeURIComponent(paramSpanValue));
			}
		}
		// create new anchor tag
		var at = String.fromCharCode(32*2);
		var email = cleanSpan(user) + at + cleanSpan(domain);
		var anchorTagText = document.createTextNode(anchorText ? anchorText : email);
		var mto = String.fromCharCode(109,97,105,108,116,111,58);
		var hrefAttr = mto + email;
			hrefAttr += paramValues.length ? '?' + paramValues.join('&') : '';
		var anchorTag = document.createElement('a');
			anchorTag.className = spamSpanMainClass;
			anchorTag.setAttribute('href', hrefAttr);
			anchorTag.appendChild(anchorTagText);
		// replace the span with anchor
		allSpamSpans[i].parentNode.replaceChild(anchorTag, allSpamSpans[i]);
	}
}

function getElementsByClass(searchClass, scope, tag) {
	var classElements = new Array();
	if (scope == null) node = document;
	if (tag == null) tag = '*';
	var els = scope.getElementsByTagName(tag);
	var elsLen = els.length;
	var pattern = new RegExp("(^|\s)"+searchClass+"(\s|$)");
	for (var i = 0, j = 0; i < elsLen; i++) {
		if ( pattern.test(els[i].className) ) {
			classElements[j] = els[i];
			j++;
		}
	}
	return classElements;
}

function getSpanValue(searchClass, scope) {
	var span = getElementsByClass(searchClass, scope, 'span');
	if (span[0]) {
		return span[0].firstChild.nodeValue;
	} else {
		return false;
	}
}

function cleanSpan(string) {
	// string = string.replace(//g, '');
	// replace variations of [dot] with .
	string = string.replace(/[\[\(\{]?[dD][oO0][tT][\}\)\]]?/g, '.');
	// replace spaces with nothing
	string = string.replace(/\s+/g, '');
	return string;
}

// http://www.quirksmode.org/blog/archives/2005/10/_and_the_winner_1.html
function addEvent(obj, type, fn) {
	if (obj.addEventListener)
		obj.addEventListener(type, fn, false);
	else if (obj.attachEvent)
	{
		obj['e' + type + fn] = fn;
		obj[type + fn] = function() { obj['e' + type + fn](window.event); }
		obj.attachEvent('on' + type, obj[type + fn]);
	}
}