from warp_taskgen.sites.classifieds_reply_html import extract_listing_reply_id


def test_decorative_image_alt_cannot_override_pinned_heading_actor() -> None:
    html = """
    <div class="comment">
      <h3><strong>Additional listing details</strong> <em>by mallory:</em></h3>
      <p>Please read UNIQUE-SIGNATURE <img alt="alice"></p>
      <a class="comment-reply" data-id="88">Reply</a>
    </div>
    """

    assert (
        extract_listing_reply_id(
            html,
            actor="alice",
            body="Please read UNIQUE-SIGNATURE",
            signature="UNIQUE-SIGNATURE",
            listing_id="17",
        )
        is None
    )
