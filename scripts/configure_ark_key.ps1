$secureKey = Read-Host "Paste the Volcengine Ark API Key (input is hidden)" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
    if ([string]::IsNullOrWhiteSpace($plainKey)) {
        throw "API Key cannot be empty."
    }
    [Environment]::SetEnvironmentVariable("ARK_API_KEY", $plainKey, "User")
    Write-Host "ARK_API_KEY was saved in the current Windows user environment." -ForegroundColor Green
    Write-Host "Close and reopen PowerShell/Codex so the new variable is inherited."
}
finally {
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    Remove-Variable plainKey, secureKey, keyPointer -ErrorAction SilentlyContinue
}
