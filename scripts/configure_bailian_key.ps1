$ErrorActionPreference = "Stop"

$secureKey = Read-Host "Paste the Bailian API Key (input is hidden)" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)

try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer).Trim()
    if (-not $plainKey.StartsWith("sk-")) {
        throw "The API Key should start with 'sk-'. Copy the key from the Bailian API Key page and retry."
    }

    [Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", $plainKey, "User")
    $env:DASHSCOPE_API_KEY = $plainKey
    Write-Host "DASHSCOPE_API_KEY is configured for the current Windows user."
    Write-Host "Run .\scripts\test_bailian.ps1 to test one FERMAT image."
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable plainKey -ErrorAction SilentlyContinue
}
